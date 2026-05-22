//! Implementation of the `z_shieldcoinbase` RPC method.

#[cfg(feature = "transparent-key-import")]
use std::collections::HashMap;
use std::collections::HashSet;
use std::convert::Infallible;
use std::future::Future;

use documented::Documented;
use jsonrpsee::core::{JsonValue, RpcResult};
use schemars::JsonSchema;
use secrecy::ExposeSecret;
use serde::Serialize;
use transparent::address::TransparentAddress;
use uuid::Uuid;
use zaino_state::FetchServiceSubscriber;
use zcash_address::ZcashAddress;
use zcash_client_backend::{
    data_api::{
        Account as _, InputSource, TransparentOutputFilter, WalletRead,
        wallet::{
            ConfirmationsPolicy, SpendingKeys, TargetHeight, create_proposed_transactions,
            input_selection::GreedyInputSelector, propose_shielding_coinbase,
        },
    },
    fees::StandardFeeRule,
    proposal::Proposal,
    wallet::OvkPolicy,
};
use zcash_client_sqlite::AccountUuid;
use zcash_keys::{address::Address, keys::UnifiedSpendingKey};
use zcash_proofs::prover::LocalTxProver;
use zcash_protocol::value::Zatoshis;

use crate::{
    components::{
        database::{DbConnection, DbHandle},
        json_rpc::{
            asyncop::{ContextInfo, OperationId},
            payments::{PrivacyPolicy, SendResult, broadcast_transactions, parse_memo},
            server::LegacyCode,
            utils::{JsonZec, value_from_zatoshis},
        },
        keystore::KeyStore,
    },
    prelude::*,
};

#[cfg(feature = "transparent-key-import")]
use zcash_client_backend::wallet::TransparentAddressSource;
#[cfg(feature = "transparent-key-import")]
use zcash_script::script;

/// Pre-flight response shape, matching `zcashd`'s `z_shieldcoinbase`:
/// `{ remainingUTXOs, remainingValue, shieldingUTXOs, shieldingValue, opid }`.
#[derive(Clone, Debug, Serialize, Documented, JsonSchema)]
pub(crate) struct ShieldCoinbaseResult {
    /// Number of coinbase UTXOs eligible for shielding that were not selected
    /// by this operation. Non-zero when the caller supplied a `limit` smaller
    /// than the count of eligible coinbase UTXOs.
    #[serde(rename = "remainingUTXOs")]
    remaining_utxos: u64,

    /// Total value (in ZEC) of the eligible-but-not-selected coinbase UTXOs.
    #[serde(rename = "remainingValue")]
    remaining_value: JsonZec,

    /// Number of coinbase UTXOs being shielded by this operation.
    #[serde(rename = "shieldingUTXOs")]
    shielding_utxos: u64,

    /// Total value (in ZEC) being shielded by this operation.
    #[serde(rename = "shieldingValue")]
    shielding_value: JsonZec,

    /// Operation id to pass to `z_getoperationstatus` /
    /// `z_getoperationresult` to retrieve the final result.
    opid: OperationId,
}

impl ShieldCoinbaseResult {
    pub(super) fn new(preflight: Preflight, opid: OperationId) -> Self {
        Self {
            remaining_utxos: preflight.remaining_utxos,
            remaining_value: preflight.remaining_value,
            shielding_utxos: preflight.shielding_utxos,
            shielding_value: preflight.shielding_value,
            opid,
        }
    }
}

pub(crate) type ResultType = ShieldCoinbaseResult;
pub(crate) type Response = RpcResult<ResultType>;

/// Pre-flight numeric fields, computed before the async portion runs.
pub(crate) struct Preflight {
    pub(super) remaining_utxos: u64,
    pub(super) remaining_value: JsonZec,
    pub(super) shielding_utxos: u64,
    pub(super) shielding_value: JsonZec,
}

pub(super) const PARAM_FROMADDRESSES_DESC: &str = "Sources of coinbase UTXOs to shield. Either an array of transparent addresses owned by this \
     wallet (all of which must belong to the same account), or a single account UUID to sweep \
     every coinbase UTXO across that account's transparent receivers.";
pub(super) const PARAM_TOADDRESS_DESC: &str = "Any Zcash shielded address (Sapling, Orchard, or Unified with a shielded receiver) that \
     will receive the shielded funds. Need not belong to this wallet. Transparent or TEX \
     destinations are rejected.";
pub(super) const PARAM_LIMIT_DESC: &str = "If supplied, caps the number of selected coinbase UTXOs to the highest-value `n` of those \
     eligible. Recommended for wallets with many eligible coinbase UTXOs: without it, a single \
     transaction is built containing all eligible UTXOs, which can exceed transaction-size \
     limits at broadcast time.";
pub(super) const PARAM_MEMO_DESC: &str = "If supplied, stored in the memo field of the resulting shielded payment. Hex-encoded, up \
     to 1024 hex characters (= 512 bytes).";

/// Soft threshold above which [`call`] emits a `warn!` log about potential
/// transaction-size issues at broadcast time.
pub(super) const COINBASE_INPUTS_WARN_THRESHOLD: u64 = 400;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn call(
    mut wallet: DbHandle,
    keystore: KeyStore,
    chain: FetchServiceSubscriber,
    fromaddresses: JsonValue,
    toaddress: String,
    limit: Option<u32>,
    memo: Option<String>,
) -> RpcResult<(
    Preflight,
    Option<ContextInfo>,
    impl Future<Output = RpcResult<SendResult>>,
)> {
    // Parse the destination address.
    //
    // `toaddress` is *not* required to be owned by this wallet — `z_shieldcoinbase`
    // is a sweep operation, and the user may legitimately want to shield directly
    // into an external recipient. The backend (`propose_shielding_coinbase`)
    // enforces that the address has a shielded receiver via
    // `ProposalError::ShieldingRequiresShieldedRecipient`; we only need to
    // produce a parseable `ZcashAddress`.
    let to_zcash_address: ZcashAddress = toaddress.parse().map_err(|_| {
        LegacyCode::InvalidParameter.with_message(format!(
            "Invalid parameter, unknown address format: {toaddress}"
        ))
    })?;

    // Parse the memo parameter (hex-encoded).
    let memo = memo.as_deref().map(parse_memo).transpose()?;
    let limit_usize = limit.map(|n| n as usize);

    // Resolve `fromaddresses` to the (single) source account + its source
    // transparent addresses.
    let (account_id, from_addrs) = resolve_fromaddresses(wallet.as_ref(), &fromaddresses)?;

    if from_addrs.is_empty() {
        return Err(LegacyCode::InvalidParameter
            .with_static("No source transparent addresses resolved from `fromaddresses`."));
    }

    let account = wallet
        .get_account(account_id)
        .map_err(|e| LegacyCode::Database.with_message(e.to_string()))?
        .ok_or_else(|| {
            LegacyCode::Database.with_message(format!("Account vanished mid-call: {account_id:?}"))
        })?;

    let confirmations_policy = APP.config().builder.confirmations_policy().map_err(|_| {
        LegacyCode::Wallet.with_message(
            "Configuration error: minimum confirmations for spending trusted TXOs \
             cannot exceed that for untrusted TXOs.",
        )
    })?;

    let params = *wallet.params();
    let input_selector = GreedyInputSelector::new();

    // Build the shielding proposal.
    let proposal = propose_shielding_coinbase::<_, _, _, _, Infallible>(
        wallet.as_mut(),
        &params,
        &input_selector,
        &StandardFeeRule::Zip317,
        Zatoshis::ZERO,
        &from_addrs,
        to_zcash_address,
        memo,
        limit_usize,
    )
    .map_err(|e| {
        LegacyCode::Wallet.with_message(format!("Failed to propose shielding transaction: {e}"))
    })?;

    // The shielding operation always reveals the transparent sender(s). No
    // other privacy weakening is required, and we deliberately don't allow
    // callers to opt into it.
    crate::components::json_rpc::payments::enforce_privacy_policy(
        &proposal,
        PrivacyPolicy::AllowRevealedSenders,
    )?;

    // Pre-flight numerics.
    let (shielding_utxos, shielding_value_zats) = sum_selected_inputs(&proposal)?;
    let target_height = proposal.min_target_height();
    let (total_utxos, total_value_zats) = enumerate_eligible(
        wallet.as_mut(),
        &from_addrs,
        target_height,
        confirmations_policy,
    )?;

    let remaining_utxos = total_utxos.checked_sub(shielding_utxos).ok_or_else(|| {
        LegacyCode::Wallet.with_static(
            "Internal accounting error: proposal selected more UTXOs than \
             enumeration found (likely a chain race during shielding setup).",
        )
    })?;
    let remaining_value_zats = (total_value_zats - shielding_value_zats).ok_or_else(|| {
        LegacyCode::Wallet.with_static(
            "Internal accounting error: proposal value exceeds enumerated total \
             (likely a chain race during shielding setup).",
        )
    })?;

    if shielding_utxos > COINBASE_INPUTS_WARN_THRESHOLD {
        warn!(
            "z_shieldcoinbase: proposal selected {} coinbase UTXOs, which exceeds the \
             soft warning threshold of {}. The resulting transaction may exceed \
             network/mempool size limits at broadcast time. If broadcast fails, retry \
             with a `limit` parameter to shield in smaller batches.",
            shielding_utxos, COINBASE_INPUTS_WARN_THRESHOLD,
        );
    }

    let preflight = Preflight {
        remaining_utxos,
        remaining_value: value_from_zatoshis(remaining_value_zats),
        shielding_utxos,
        shielding_value: value_from_zatoshis(shielding_value_zats),
    };

    // Derive the spending key for the source account.
    let derivation = account.source().key_derivation().ok_or_else(|| {
        LegacyCode::InvalidAddressOrKey.with_message(format!(
            "No payment source found for account {}.",
            account_id.expose_uuid(),
        ))
    })?;
    let seed = keystore
        .decrypt_seed(derivation.seed_fingerprint())
        .await
        .map_err(|e| match e.kind() {
            crate::error::ErrorKind::Generic if e.to_string() == "Wallet is locked" => {
                LegacyCode::WalletUnlockNeeded.with_message(e.to_string())
            }
            _ => LegacyCode::Database.with_message(e.to_string()),
        })?;
    let usk = UnifiedSpendingKey::from_seed(
        wallet.params(),
        seed.expose_secret(),
        derivation.account_index(),
    )
    .map_err(|e| LegacyCode::InvalidAddressOrKey.with_message(e.to_string()))?;

    #[cfg(feature = "transparent-key-import")]
    let standalone_keys =
        collect_standalone_keys(wallet.as_mut(), &keystore, account_id, &proposal).await?;

    Ok((
        preflight,
        Some(ContextInfo::new(
            "z_shieldcoinbase",
            serde_json::json!({
                "fromaddresses": fromaddresses,
                "toaddress": toaddress,
                "limit": limit,
            }),
        )),
        run(
            wallet,
            chain,
            proposal,
            SpendingKeys::new(
                usk,
                #[cfg(feature = "transparent-key-import")]
                standalone_keys,
            ),
        ),
    ))
}

/// Resolve `fromaddresses` to a single source account and the transparent
/// addresses to draw coinbase UTXOs from.
///
/// Accepted shapes:
/// * a single account UUID (string) — sweeps every transparent receiver in
///   that account.
/// * an array of transparent addresses, all of which must belong to the same
///   wallet-owned account.
fn resolve_fromaddresses(
    wallet: &DbConnection,
    fromaddresses: &JsonValue,
) -> RpcResult<(AccountUuid, Vec<TransparentAddress>)> {
    if let Some(s) = fromaddresses.as_str() {
        let uuid = Uuid::parse_str(s).map_err(|_| {
            LegacyCode::InvalidParameter.with_message(format!(
                "Invalid `fromaddresses`: expected an account UUID string or an array of \
                 transparent addresses, got {s:?}.",
            ))
        })?;
        let account_id = AccountUuid::from_uuid(uuid);
        if wallet
            .get_account(account_id)
            .map_err(|e| LegacyCode::Database.with_message(e.to_string()))?
            .is_none()
        {
            return Err(
                LegacyCode::InvalidParameter.with_message(format!("Unknown account UUID: {s}"))
            );
        }
        let from_addrs = wallet
            .get_transparent_receivers(account_id, true, true)
            .map_err(|e| LegacyCode::Database.with_message(e.to_string()))?
            .into_keys()
            .collect();
        return Ok((account_id, from_addrs));
    }

    let arr = fromaddresses.as_array().ok_or_else(|| {
        LegacyCode::InvalidParameter.with_static(
            "Invalid `fromaddresses`: expected an account UUID string or an array of \
             transparent addresses.",
        )
    })?;
    if arr.is_empty() {
        return Err(LegacyCode::InvalidParameter
            .with_static("Invalid `fromaddresses`: array must not be empty."));
    }

    let mut seen: HashSet<TransparentAddress> = HashSet::new();
    let mut from_addrs: Vec<TransparentAddress> = Vec::with_capacity(arr.len());
    let mut account_id: Option<AccountUuid> = None;

    for item in arr {
        let s = item.as_str().ok_or_else(|| {
            LegacyCode::InvalidParameter
                .with_static("Invalid `fromaddresses`: every array entry must be a string.")
        })?;
        let address = Address::decode(wallet.params(), s).ok_or_else(|| {
            LegacyCode::InvalidAddressOrKey.with_message(format!(
                "Invalid `fromaddresses` entry: not a Zcash address: {s}"
            ))
        })?;
        let transparent_addr = match address {
            Address::Transparent(addr) => addr,
            _ => {
                return Err(LegacyCode::InvalidAddressOrKey.with_message(format!(
                    "Invalid `fromaddresses` entry: only transparent addresses are accepted: {s}",
                )));
            }
        };

        if !seen.insert(transparent_addr) {
            continue;
        }

        let owner = wallet
            .find_account_for_address(wallet.params(), &Address::Transparent(transparent_addr))
            .map_err(|e| LegacyCode::Database.with_message(e.to_string()))?
            .ok_or_else(|| {
                LegacyCode::InvalidAddressOrKey.with_message(format!(
                    "Transparent address is not owned by any account in this wallet: {s}",
                ))
            })?;

        match account_id {
            None => account_id = Some(owner),
            Some(existing) if existing == owner => {}
            Some(_) => {
                return Err(LegacyCode::InvalidParameter.with_static(
                    "All addresses in `fromaddresses` must belong to the same account.",
                ));
            }
        }
        from_addrs.push(transparent_addr);
    }

    Ok((account_id.expect("array is non-empty"), from_addrs))
}

fn sum_selected_inputs(
    proposal: &Proposal<StandardFeeRule, Infallible>,
) -> RpcResult<(u64, Zatoshis)> {
    let mut count: u64 = 0;
    let mut sum = Zatoshis::ZERO;
    for step in proposal.steps() {
        for utxo in step.transparent_inputs() {
            count = count.saturating_add(1);
            sum = (sum + utxo.value()).ok_or_else(|| {
                LegacyCode::Wallet
                    .with_static("Internal error: shielding value sum overflowed Zatoshis bounds.")
            })?;
        }
    }
    Ok((count, sum))
}

fn enumerate_eligible(
    wallet: &mut DbConnection,
    from_addrs: &[TransparentAddress],
    target_height: TargetHeight,
    confirmations_policy: ConfirmationsPolicy,
) -> RpcResult<(u64, Zatoshis)> {
    let mut total_utxos: u64 = 0;
    let mut total_value_zats = Zatoshis::ZERO;
    for addr in from_addrs {
        let utxos = wallet
            .get_spendable_transparent_outputs(
                addr,
                target_height,
                confirmations_policy,
                TransparentOutputFilter::CoinbaseOnly,
            )
            .map_err(|e| LegacyCode::Database.with_message(e.to_string()))?;
        total_utxos = total_utxos.saturating_add(utxos.len() as u64);
        for utxo in utxos {
            total_value_zats = (total_value_zats + utxo.value()).ok_or_else(|| {
                LegacyCode::Wallet.with_static(
                    "Internal error: total transparent value overflowed Zatoshis bounds.",
                )
            })?;
        }
    }
    Ok((total_utxos, total_value_zats))
}

#[cfg(feature = "transparent-key-import")]
async fn collect_standalone_keys(
    wallet: &mut DbConnection,
    keystore: &KeyStore,
    account_id: AccountUuid,
    proposal: &Proposal<StandardFeeRule, Infallible>,
) -> RpcResult<HashMap<TransparentAddress, Vec<secp256k1::SecretKey>>> {
    let standalone_addrs: HashSet<TransparentAddress> = wallet
        .get_transparent_receivers(account_id, true, true)
        .map_err(|e| LegacyCode::Database.with_message(e.to_string()))?
        .into_iter()
        .filter_map(|(addr, metadata)| match metadata.source() {
            TransparentAddressSource::StandalonePubkey(_)
            | TransparentAddressSource::StandaloneScript(_) => Some(addr),
            TransparentAddressSource::Derived { .. } => None,
        })
        .collect();

    let mut keys: HashMap<TransparentAddress, Vec<secp256k1::SecretKey>> = HashMap::new();
    for step in proposal.steps() {
        for input in step.transparent_inputs() {
            if let Some(address) = script::FromChain::parse(&input.txout().script_pubkey().0)
                .ok()
                .as_ref()
                .and_then(TransparentAddress::from_script_from_chain)
            {
                if !standalone_addrs.contains(&address) {
                    continue;
                }
                let secret_key = keystore
                    .decrypt_standalone_transparent_key(&address)
                    .await
                    .map_err(|e| match e.kind() {
                        crate::error::ErrorKind::Generic if e.to_string() == "Wallet is locked" => {
                            LegacyCode::WalletUnlockNeeded.with_message(e.to_string())
                        }
                        _ => LegacyCode::Database.with_message(e.to_string()),
                    })?;
                keys.entry(address).or_default().push(secret_key);
            }
        }
    }
    Ok(keys)
}

/// Construct and broadcast the shielding transaction.
async fn run(
    mut wallet: DbHandle,
    chain: FetchServiceSubscriber,
    proposal: Proposal<StandardFeeRule, Infallible>,
    spending_keys: SpendingKeys,
) -> RpcResult<SendResult> {
    let prover = LocalTxProver::bundled();
    let (wallet, txids) = crate::spawn_blocking!("z_shieldcoinbase runner", move || {
        let params = *wallet.params();
        create_proposed_transactions::<_, _, Infallible, _, Infallible, _>(
            wallet.as_mut(),
            &params,
            &prover,
            &prover,
            &spending_keys,
            OvkPolicy::Sender,
            &proposal,
        )
        .map(|txids| (wallet, txids))
    })
    .await
    .map_err(|e| {
        LegacyCode::Wallet.with_message(format!("Failed to build shielding transaction: {e}"))
    })?
    .map_err(|e| {
        LegacyCode::Wallet.with_message(format!("Failed to build shielding transaction: {e}"))
    })?;

    broadcast_transactions(&wallet, chain, txids.into()).await
}
