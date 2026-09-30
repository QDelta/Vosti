// Executable request state plus structural request semantics.  Dafny `Seq`
// is executable, but Verus `Seq` is spec-only, so token histories are stored
// as `Vec<TokenId>` and specs use the `@` view.

pub use crate::boundary::sampler::SamplerState;
use crate::types::{RequestId, TokenId};
use vstd::assert_sets_equal;
use vstd::prelude::*;

verus! {

// Equality of EOS policies is intentionally opaque outside this module.  Set
// extensionality is useful at semantic boundaries, but eagerly expanding it
// inside every open request-machine transition makes scheduler loops pay for
// the backing representation.  Callers can cross this boundary explicitly
// through the small bridge lemmas below.
pub closed spec fn same_eos_tokens(a: Set<TokenId>, b: Set<TokenId>) -> bool {
    a == b
}

pub proof fn lemma_same_eos_tokens_from_view_eq(
    a: Set<TokenId>,
    b: Set<TokenId>,
)
    requires
        a == b,
    ensures
        same_eos_tokens(a, b),
{
    reveal(same_eos_tokens);
}

pub proof fn lemma_same_eos_tokens_view_eq(
    a: Set<TokenId>,
    b: Set<TokenId>,
)
    requires
        same_eos_tokens(a, b),
    ensures
        a == b,
{
    reveal(same_eos_tokens);
}

pub proof fn lemma_same_eos_tokens_transitive(
    a: Set<TokenId>,
    b: Set<TokenId>,
    c: Set<TokenId>,
)
    requires
        same_eos_tokens(a, b),
        same_eos_tokens(b, c),
    ensures
        same_eos_tokens(a, c),
{
    reveal(same_eos_tokens);
}

pub proof fn lemma_same_eos_tokens_contains(
    a: Set<TokenId>,
    b: Set<TokenId>,
    token: TokenId,
)
    requires
        same_eos_tokens(a, b),
    ensures
        a.contains(token) == b.contains(token),
{
    reveal(same_eos_tokens);
}

pub proof fn lemma_eos_tokens_from_policy_eq(a: RequestState, b: RequestState)
    requires
        a.eos_token_1 == b.eos_token_1,
        a.eos_token_2 == b.eos_token_2,
        a.eos_token_3 == b.eos_token_3,
    ensures
        same_eos_tokens(eos_tokens(a), eos_tokens(b)),
{
    reveal(eos_tokens);
    lemma_same_eos_tokens_from_view_eq(
        eos_tokens(a),
        eos_tokens(b),
    );
}

// The supported Qwen 3 and Gemma 3 checkpoints, plus the target Llama 3.1
// checkpoint, use at most three alternative one-token terminators. Encoding that
// admitted bound as a finite value type keeps the executable request state
// free of a sequence-valued field and gives the solver a cheap structural
// equality.  The private fields make this an immutable value outside this
// module; unused slots repeat the first token, so every value is nonempty.
// Fixed implementation bound, not a tunable limit. The Python counterpart is in
// python/vosti_kernels/serving_workload.py. Changing both values is insufficient:
// review/update the three-field representation, lifecycle logic, and proofs,
// then reverify and retest.
pub const MAX_EOS_TOKEN_IDS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EosTokenSet {
    first: TokenId,
    second: TokenId,
    third: TokenId,
}

pub closed spec fn eos_token_set_view(policy: EosTokenSet) -> Set<TokenId> {
    Set::<TokenId>::empty()
        .insert(policy.first)
        .insert(policy.second)
        .insert(policy.third)
}

impl EosTokenSet {
    pub fn singleton(token: TokenId) -> (out: Self)
        ensures
            eos_token_set_view(out) == Set::<TokenId>::empty().insert(token),
    {
        let out = EosTokenSet {
            first: token,
            second: token,
            third: token,
        };
        proof { reveal(eos_token_set_view); }
        out
    }

    pub fn from_nonempty_bounded(ids: &Vec<TokenId>) -> (out: Self)
        requires
            0 < ids@.len(),
            ids@.len() <= MAX_EOS_TOKEN_IDS as int,
        ensures
            same_eos_tokens(eos_token_set_view(out), ids@.to_set()),
    {
        let out = if ids.len() == 1 {
            EosTokenSet {
                first: ids[0],
                second: ids[0],
                third: ids[0],
            }
        } else if ids.len() == 2 {
            EosTokenSet {
                first: ids[0],
                second: ids[1],
                third: ids[0],
            }
        } else {
            EosTokenSet {
                first: ids[0],
                second: ids[1],
                third: ids[2],
            }
        };
        proof {
            reveal(eos_token_set_view);
            assert_sets_equal!(eos_token_set_view(out) == ids@.to_set(), token: TokenId => {
                if ids@.to_set().contains(token) {
                    assert(ids@.contains(token));
                    let i = choose|i: int| 0 <= i < ids@.len()
                        && ids@[i] == token;
                    assert(i == 0 || i == 1 || i == 2);
                }
                if eos_token_set_view(out).contains(token) {
                    if ids.len() == 1 {
                        assert(ids@[0] == token);
                    } else if ids.len() == 2 {
                        assert(ids@[0] == token || ids@[1] == token);
                    } else {
                        assert(ids@[0] == token || ids@[1] == token
                            || ids@[2] == token);
                    }
                }
            });
            lemma_same_eos_tokens_from_view_eq(
                eos_token_set_view(out), ids@.to_set(),
            );
        }
        out
    }

    pub fn contains(&self, token: TokenId) -> (out: bool)
        ensures
            out == eos_token_set_view(*self).contains(token),
    {
        reveal(eos_token_set_view);
        token == self.first || token == self.second || token == self.third
    }
}

// Runtime-facing request admission input.  Generated tokens and sampler state
// are intentionally not caller-controlled: online arrivals always enter in the
// same fresh phase as startup requests.
pub struct NewRequest {
    pub request_id: RequestId,
    pub prompt_tokens: Vec<TokenId>,
    pub max_tokens: usize,
    // Any member terminates generation.  This is a set of alternative
    // one-token terminators, not a multi-token stop sequence.
    pub eos_token_ids: Vec<TokenId>,
    pub ignore_eos: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum AdmissionStatus {
    Accepted,
    DuplicateRequestId,
    EmptyPrompt,
    EmptyEosTokenIds,
    TooManyEosTokenIds,
    ZeroMaxTokens,
    HistoryCapacityOverflow,
}

pub struct RequestState {
    pub request_id: RequestId,
    pub prompt_tokens: Vec<TokenId>,
    pub generated_tokens: Vec<TokenId>,
    pub sampler_state: SamplerState,
    pub max_tokens: usize,
    // Fixed-width executable termination policy. Unused slots repeat token 1,
    // so this always denotes a nonempty set without putting a Vec or nested
    // datatype into scheduler structural equalities.
    pub eos_token_1: TokenId,
    pub eos_token_2: TokenId,
    pub eos_token_3: TokenId,
    pub ignore_eos: bool,
}

pub closed spec fn eos_tokens(s: RequestState) -> Set<TokenId> {
    Set::<TokenId>::empty()
        .insert(s.eos_token_1)
        .insert(s.eos_token_2)
        .insert(s.eos_token_3)
}

// Request-lifecycle controls are operationally important, but invisible to
// model evaluation and sampling.  Keep their equality behind a closed
// predicate so scheduler loops can frame the policy as one fact instead of
// repeatedly expanding every scalar slot.  Proofs that decide termination or
// establish strong engine/IBM coherence cross this boundary explicitly.
pub closed spec fn request_lifecycle_view_eq(
    a: RequestState,
    b: RequestState,
) -> bool {
    a.request_id == b.request_id
    && a.max_tokens == b.max_tokens
    && a.eos_token_1 == b.eos_token_1
    && a.eos_token_2 == b.eos_token_2
    && a.eos_token_3 == b.eos_token_3
    && a.ignore_eos == b.ignore_eos
}

pub proof fn lemma_request_lifecycle_view_eq_reflexive(s: RequestState)
    ensures
        request_lifecycle_view_eq(s, s),
{
    reveal(request_lifecycle_view_eq);
}

pub proof fn lemma_request_lifecycle_view_eq_from_fields(
    a: RequestState,
    b: RequestState,
)
    requires
        a.request_id == b.request_id,
        a.max_tokens == b.max_tokens,
        a.eos_token_1 == b.eos_token_1,
        a.eos_token_2 == b.eos_token_2,
        a.eos_token_3 == b.eos_token_3,
        a.ignore_eos == b.ignore_eos,
    ensures
        request_lifecycle_view_eq(a, b),
{
    reveal(request_lifecycle_view_eq);
}

pub proof fn lemma_request_lifecycle_view_eq_fields(
    a: RequestState,
    b: RequestState,
)
    requires
        request_lifecycle_view_eq(a, b),
    ensures
        a.request_id == b.request_id,
        a.max_tokens == b.max_tokens,
        a.eos_token_1 == b.eos_token_1,
        a.eos_token_2 == b.eos_token_2,
        a.eos_token_3 == b.eos_token_3,
        a.ignore_eos == b.ignore_eos,
{
    reveal(request_lifecycle_view_eq);
}

pub proof fn lemma_request_lifecycle_view_eq_symmetric(
    a: RequestState,
    b: RequestState,
)
    requires
        request_lifecycle_view_eq(a, b),
    ensures
        request_lifecycle_view_eq(b, a),
{
    reveal(request_lifecycle_view_eq);
}

pub proof fn lemma_request_lifecycle_view_eq_transitive(
    a: RequestState,
    b: RequestState,
    c: RequestState,
)
    requires
        request_lifecycle_view_eq(a, b),
        request_lifecycle_view_eq(b, c),
    ensures
        request_lifecycle_view_eq(a, c),
{
    reveal(request_lifecycle_view_eq);
}

impl RequestState {
    pub fn from_parts(
        request_id: RequestId,
        prompt_tokens: Vec<TokenId>,
        generated_tokens: Vec<TokenId>,
        sampler_state: SamplerState,
        max_tokens: usize,
        eos_token_set: EosTokenSet,
        ignore_eos: bool,
    ) -> (out: Self)
        ensures
            out.request_id == request_id,
            out.prompt_tokens@ == prompt_tokens@,
            out.generated_tokens@ == generated_tokens@,
            out.sampler_state == sampler_state,
            out.max_tokens == max_tokens,
            same_eos_tokens(eos_tokens(out), eos_token_set_view(eos_token_set)),
            out.ignore_eos == ignore_eos,
    {
        RequestState {
            request_id,
            prompt_tokens,
            generated_tokens,
            sampler_state,
            max_tokens,
            eos_token_1: eos_token_set.first,
            eos_token_2: eos_token_set.second,
            eos_token_3: eos_token_set.third,
            ignore_eos,
        }
    }

    pub fn eos_contains(&self, token: TokenId) -> (out: bool)
        ensures
            out == eos_tokens(*self).contains(token),
    {
        reveal(eos_tokens);
        token == self.eos_token_1
            || token == self.eos_token_2
            || token == self.eos_token_3
    }
}

impl Clone for RequestState {
    fn clone(&self) -> (out: Self)
        ensures
            out.request_id == self.request_id,
            out.prompt_tokens@ == self.prompt_tokens@,
            out.generated_tokens@ == self.generated_tokens@,
            out.sampler_state == self.sampler_state,
            out.max_tokens == self.max_tokens,
            out.eos_token_1 == self.eos_token_1,
            out.eos_token_2 == self.eos_token_2,
            out.eos_token_3 == self.eos_token_3,
            out.ignore_eos == self.ignore_eos,
            valid_request_state(out) == valid_request_state(*self),
            is_finished(out) == is_finished(*self),
            can_step(out) == can_step(*self),
    {
        RequestState {
            request_id: self.request_id,
            prompt_tokens: self.prompt_tokens.clone(),
            generated_tokens: self.generated_tokens.clone(),
            sampler_state: self.sampler_state,
            max_tokens: self.max_tokens,
            eos_token_1: self.eos_token_1,
            eos_token_2: self.eos_token_2,
            eos_token_3: self.eos_token_3,
            ignore_eos: self.ignore_eos,
        }
    }
}

// Strong operational equality used inside one engine/IBM refinement.  It
// includes lifecycle controls because those two machines must remove the same
// requests, but compares executable token histories through their Seq views.
pub open spec fn request_state_view_eq(a: RequestState, b: RequestState) -> bool {
    request_lifecycle_view_eq(a, b)
    && a.prompt_tokens@ == b.prompt_tokens@
    && a.generated_tokens@ == b.generated_tokens@
    && a.sampler_state == b.sampler_state
}

// The model and sampler cannot observe request lifecycle policy.  This is the
// deliberately weaker relation used by cross-execution output determinism:
// EOS, ignore_eos, max_tokens, and request_id may differ.
pub open spec fn request_sampling_view_eq(a: RequestState, b: RequestState) -> bool {
    a.prompt_tokens@ == b.prompt_tokens@
    && a.generated_tokens@ == b.generated_tokens@
    && a.sampler_state == b.sampler_state
}

pub proof fn lemma_request_state_view_eq_reflexive(s: RequestState)
    ensures
        request_state_view_eq(s, s),
{
    lemma_request_lifecycle_view_eq_reflexive(s);
}

pub open spec fn valid_request_state(s: RequestState) -> bool {
    s.prompt_tokens@.len() > 0
    && s.generated_tokens@.len() <= s.max_tokens as int
}

pub open spec fn is_finished(s: RequestState) -> bool {
    s.generated_tokens@.len() == s.max_tokens as int
        || (
            !s.ignore_eos
            && 0 < s.generated_tokens@.len()
            && eos_tokens(s).contains(
                s.generated_tokens@[s.generated_tokens@.len() - 1],
            )
        )
}

pub open spec fn can_step(s: RequestState) -> bool {
    valid_request_state(s) && !is_finished(s)
}

// Static per-request capacity budget used by repeated executable serving.
// `max_tokens` bounds every reachable generated-token length, so bounding the
// prompt plus that maximum once is enough to keep both Rust `usize` history
// arithmetic and the scheduler's u64 cumulative-key arithmetic representable
// at every surviving step.
pub open spec fn request_history_capacity_safe(s: RequestState) -> bool {
    s.prompt_tokens@.len() + s.max_tokens as int <= usize::MAX as int
    && s.prompt_tokens@.len() + s.max_tokens as int <= u64::MAX as int
}

// Admissible fresh workload for repeated serving. Pairwise-distinct ids make
// initialization a map embedding, and the static history budget remains
// inductive across every surviving step.
pub open spec fn initial_request_batch_ready(requests: Seq<RequestState>) -> bool {
    (forall|a: int, b: int|
        #![trigger requests[a].request_id, requests[b].request_id]
        0 <= a < b < requests.len() ==>
            requests[a].request_id != requests[b].request_id)
    && (forall|k: int| 0 <= k < requests.len() ==> {
        let state = #[trigger] requests[k];
        &&& state.generated_tokens@.len() == 0
        &&& can_step(state)
        &&& request_history_capacity_safe(state)
    })
}

pub open spec fn history(s: RequestState) -> Seq<TokenId>
    recommends valid_request_state(s),
{
    s.prompt_tokens@ + s.generated_tokens@
}

pub open spec fn num_tokens(s: RequestState) -> nat
    recommends valid_request_state(s),
{
    (s.prompt_tokens@.len() + s.generated_tokens@.len()) as nat
}

pub open spec fn should_finish_after_append(s: RequestState, emitted: TokenId) -> bool
{
    s.generated_tokens@.len() + 1 == s.max_tokens as int
        || (!s.ignore_eos && eos_tokens(s).contains(emitted))
}

pub fn should_finish_after_append_exec(s: &RequestState, emitted: TokenId) -> (out: bool)
    requires
        can_step(*s),
        s.generated_tokens@.len() < usize::MAX as int,
    ensures
        out == should_finish_after_append(*s, emitted),
{
    let next_len = s.generated_tokens.len() + 1;
    assert(next_len as int == s.generated_tokens@.len() + 1);
    if next_len == s.max_tokens {
        assert(s.generated_tokens@.len() + 1 == s.max_tokens as int);
        true
    } else if !s.ignore_eos && s.eos_contains(emitted) {
        true
    } else {
        assert(s.generated_tokens@.len() + 1 != s.max_tokens as int);
        assert(s.ignore_eos || !eos_tokens(*s).contains(emitted));
        false
    }
}

pub fn history_len(s: &RequestState) -> (out: usize)
    requires
        valid_request_state(*s),
        s.prompt_tokens@.len() + s.generated_tokens@.len() <= usize::MAX as int,
    ensures
        out as int == history(*s).len(),
{
    let out = s.prompt_tokens.len() + s.generated_tokens.len();
    assert(out as int == s.prompt_tokens@.len() + s.generated_tokens@.len());
    out
}

pub fn last_history_token(s: &RequestState) -> (out: TokenId)
    requires
        valid_request_state(*s),
    ensures
        out == history(*s)[history(*s).len() - 1],
{
    if s.generated_tokens.len() > 0 {
        let i = s.generated_tokens.len() - 1;
        assert(i as int == s.generated_tokens@.len() - 1);
        let out = s.generated_tokens[i];
        assert(history(*s).len() == s.prompt_tokens@.len() + s.generated_tokens@.len());
        assert(history(*s)[history(*s).len() - 1] == s.generated_tokens@[i as int]);
        out
    } else {
        assert(s.prompt_tokens@.len() > 0);
        let i = s.prompt_tokens.len() - 1;
        assert(i as int == s.prompt_tokens@.len() - 1);
        let out = s.prompt_tokens[i];
        assert(s.generated_tokens@.len() == 0);
        assert(history(*s) == s.prompt_tokens@);
        out
    }
}

pub fn append_generated_token(
    s: &RequestState,
    next_sampler_state: SamplerState,
    token: TokenId,
) -> (out: RequestState)
    requires
        valid_request_state(*s),
        s.generated_tokens@.len() < usize::MAX as int,
    ensures
        out.request_id == s.request_id,
        out.prompt_tokens@ == s.prompt_tokens@,
        out.generated_tokens@ == s.generated_tokens@.push(token),
        out.sampler_state == next_sampler_state,
        out.max_tokens == s.max_tokens,
        out.eos_token_1 == s.eos_token_1,
        out.eos_token_2 == s.eos_token_2,
        out.eos_token_3 == s.eos_token_3,
        out.ignore_eos == s.ignore_eos,
{
    let mut generated_tokens = s.generated_tokens.clone();
    generated_tokens.push(token);
    RequestState {
        request_id: s.request_id,
        prompt_tokens: s.prompt_tokens.clone(),
        generated_tokens,
        sampler_state: next_sampler_state,
        max_tokens: s.max_tokens,
        eos_token_1: s.eos_token_1,
        eos_token_2: s.eos_token_2,
        eos_token_3: s.eos_token_3,
        ignore_eos: s.ignore_eos,
    }
}

} // verus!
