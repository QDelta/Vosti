//! Source-line inventory, not a proof checker or a semantic erasure pass.
use proc_macro2::{Span, TokenStream, TokenTree};
use quote::ToTokens;
use std::{collections::BTreeSet, env, fs};
use verus_syn::{
    self as syn,
    spanned::Spanned,
    visit::{self, Visit},
};

// Each non-comment source line has one category. Specific syntax overrides
// enclosing proof functions; excluded documentation/test items override all.
const IMPL: u8 = 0;
const PROOF: u8 = 1;
const SPEC: u8 = 2;
const EXCLUDE: u8 = 3;

// Declaration extents only; dependency identities come from typed Verus VIR.
#[derive(Default)]
struct Surface {
    units: Vec<serde_json::Value>,
}
impl Surface {
    fn record<T: ToTokens>(&mut self, name: &str, node: &T, kind: &str, assumption: bool) {
        self.units.push(serde_json::json!({
            "name": name, "start": node.span().start().line,
            "end": node.span().end().line, "kind": kind, "assumption": assumption,
        }));
    }
    fn function<T: ToTokens>(&mut self, node: &T, sig: &syn::Signature,
                            attrs: &[syn::Attribute]) {
        let spec = matches!(sig.mode, syn::FnMode::Spec(_) | syn::FnMode::SpecChecked(_));
        let proof = matches!(sig.mode, syn::FnMode::Proof(_) | syn::FnMode::ProofAxiom(_));
        let text = sig.to_token_stream().to_string();
        let assumption = text.split_whitespace().any(|word| word == "uninterp")
            || matches!(sig.mode, syn::FnMode::ProofAxiom(_))
            || attrs.iter().any(|a|
                a.to_token_stream().to_string().replace(' ', "") == "#[verifier::external_body]");
        let extent = if spec || (proof && assumption) { node.to_token_stream() } else {
            sig.to_token_stream()
        };
        self.record(&sig.ident.to_string(), &extent,
            if spec { "spec" } else if proof { "proof_contract" } else { "exec_contract" },
            assumption);
    }
}
impl<'ast> Visit<'ast> for Surface {
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.function(item, &item.sig, &item.attrs);
    }
    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.function(item, &item.sig, &item.attrs);
    }
    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        self.function(item, &item.sig, &item.attrs);
    }
    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        self.record(&item.ident.to_string(), item, "type", false);
    }
    fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
        self.record(&item.ident.to_string(), item, "type", false);
    }
    fn visit_item_type(&mut self, item: &'ast syn::ItemType) {
        self.record(&item.ident.to_string(), item, "type", false);
    }
    fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
        self.record(&item.ident.to_string(), item, "constant", false);
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if mac.path.is_ident("verus") {
            self.visit_file(&syn::parse2(mac.tokens.clone()).expect("parse verus! surface"));
        }
    }
}

struct Counter {
    lines: Vec<u8>,
    code: BTreeSet<usize>,
    trusted: Vec<serde_json::Value>,
}

impl Counter {
    fn mark_span(&mut self, span: Span, category: u8) {
        for line in span.start().line..=span.end().line {
            if line > 0 && line <= self.lines.len() {
                self.lines[line - 1] = self.lines[line - 1].max(category);
            }
        }
    }
    fn mark<T: ToTokens>(&mut self, node: &T, category: u8) {
        if !node.to_token_stream().is_empty() {
            self.mark_span(node.span(), category);
        }
    }
    fn tokens(&mut self, stream: TokenStream) {
        for token in stream {
            match token {
                TokenTree::Group(g) => {
                    self.code.insert(g.span_open().start().line);
                    self.code.insert(g.span_close().end().line);
                    self.tokens(g.stream());
                }
                token => {
                    self.code
                        .extend(token.span().start().line..=token.span().end().line);
                }
            }
        }
    }
    fn attrs_skip<T: ToTokens>(&mut self, attrs: &[syn::Attribute], item: &T) -> bool {
        let skip = attrs.iter().any(|a| {
            let s = a.to_token_stream().to_string().replace(' ', "");
            s == "#[cfg(any())]" || s == "#[cfg(test)]" || s == "#[test]"
        });
        if skip {
            self.mark(item, EXCLUDE);
        }
        skip
    }
    fn function<T: ToTokens>(
        &mut self,
        node: &T,
        sig: &syn::Signature,
        attrs: &[syn::Attribute],
    ) -> bool {
        if self.attrs_skip(attrs, node) {
            return true;
        }
        let external = attrs.iter().any(|a| {
            a.to_token_stream().to_string().replace(' ', "") == "#[verifier::external_body]"
        });
        let proof = matches!(sig.mode, syn::FnMode::Proof(_) | syn::FnMode::ProofAxiom(_));
        if external || matches!(sig.mode, syn::FnMode::ProofAxiom(_)) {
            self.trusted.push(serde_json::json!({"name":sig.ident.to_string(),
                "line":sig.ident.span().start().line,"kind":if proof {"assumed_contract"} else {"external_body"}}));
        }
        if matches!(sig.mode, syn::FnMode::Spec(_) | syn::FnMode::SpecChecked(_))
            || (proof && external)
            || matches!(sig.mode, syn::FnMode::ProofAxiom(_))
        {
            self.mark(node, SPEC);
            // Visit attrs to remove doc comments from the source-line inventory.
            for a in attrs {
                self.visit_attribute(a);
            }
            return true;
        }
        if proof {
            self.mark(node, PROOF);
        }
        false
    }
}

impl<'ast> Visit<'ast> for Counter {
    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        if !self.function(i, &i.sig, &i.attrs) {
            visit::visit_item_fn(self, i);
        }
    }
    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        if !self.function(i, &i.sig, &i.attrs) {
            visit::visit_impl_item_fn(self, i);
        }
    }
    fn visit_trait_item_fn(&mut self, i: &'ast syn::TraitItemFn) {
        if !self.function(i, &i.sig, &i.attrs) {
            visit::visit_trait_item_fn(self, i);
        }
    }
    fn visit_signature_spec(&mut self, i: &'ast syn::SignatureSpec) {
        self.mark(i, SPEC);
    }
    fn visit_requires(&mut self, i: &'ast syn::Requires) {
        self.mark(i, SPEC);
    }
    fn visit_ensures(&mut self, i: &'ast syn::Ensures) {
        self.mark(i, SPEC);
    }
    fn visit_decreases(&mut self, i: &'ast syn::Decreases) {
        self.mark(i, SPEC);
    }
    fn visit_invariant(&mut self, i: &'ast syn::Invariant) {
        self.mark(i, SPEC);
    }
    fn visit_invariant_ensures(&mut self, i: &'ast syn::InvariantEnsures) {
        self.mark(i, SPEC);
    }
    fn visit_invariant_except_break(&mut self, i: &'ast syn::InvariantExceptBreak) {
        self.mark(i, SPEC);
    }
    fn visit_loop_spec(&mut self, i: &'ast syn::LoopSpec) {
        self.mark(&i.invariants, SPEC);
        self.mark(&i.invariant_except_breaks, SPEC);
        self.mark(&i.ensures, SPEC);
        self.mark(&i.decreases, SPEC);
    }
    fn visit_local(&mut self, i: &'ast syn::Local) {
        if i.ghost.is_some() || i.tracked.is_some() {
            self.mark(i, PROOF);
        }
        visit::visit_local(self, i);
    }
    fn visit_field(&mut self, i: &'ast syn::Field) {
        if matches!(i.mode, syn::DataMode::Ghost(_) | syn::DataMode::Tracked(_))
            || i.ty.to_token_stream().to_string().starts_with("Ghost <")
            || i.ty.to_token_stream().to_string().starts_with("Tracked <")
        {
            self.mark(i, SPEC);
        }
        visit::visit_field(self, i);
    }
    fn visit_fn_arg(&mut self, i: &'ast syn::FnArg) {
        if i.tracked.is_some() {
            self.mark(i, SPEC);
        }
        visit::visit_fn_arg(self, i);
    }
    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        if self.attrs_skip(&i.attrs, i) {
            return;
        }
        if matches!(i.mode, syn::DataMode::Ghost(_) | syn::DataMode::Tracked(_)) {
            self.mark(i, SPEC);
        }
        visit::visit_item_struct(self, i);
    }
    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        if self.attrs_skip(&i.attrs, i) {
            return;
        }
        if matches!(i.mode, syn::DataMode::Ghost(_) | syn::DataMode::Tracked(_)) {
            self.mark(i, SPEC);
        }
        visit::visit_item_enum(self, i);
    }
    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        if matches!(i.mode, syn::FnMode::Spec(_) | syn::FnMode::SpecChecked(_)) {
            self.mark(i, SPEC);
        }
        visit::visit_item_const(self, i);
    }
    fn visit_impl_item_const(&mut self, i: &'ast syn::ImplItemConst) {
        if matches!(i.mode, syn::FnMode::Spec(_) | syn::FnMode::SpecChecked(_)) {
            self.mark(i, SPEC);
        }
        visit::visit_impl_item_const(self, i);
    }
    fn visit_trait_item_const(&mut self, i: &'ast syn::TraitItemConst) {
        if matches!(i.mode, syn::FnMode::Spec(_) | syn::FnMode::SpecChecked(_)) {
            self.mark(i, SPEC);
        }
        visit::visit_trait_item_const(self, i);
    }
    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        if !self.attrs_skip(&i.attrs, i) {
            visit::visit_item_mod(self, i);
        }
    }
    fn visit_expr(&mut self, i: &'ast syn::Expr) {
        match i {
            syn::Expr::Assert(_) | syn::Expr::AssertForall(_) | syn::Expr::RevealHide(_) => {
                self.mark(i, PROOF)
            }
            syn::Expr::Assume(_) => {
                self.mark(i, SPEC);
                self.trusted
                    .push(serde_json::json!({"kind":"assume", "line":i.span().start().line,
                                           "end":i.span().end().line}));
            }
            syn::Expr::Unary(u) if matches!(u.op, syn::UnOp::Proof(_)) => self.mark(i, PROOF),
            _ => (),
        }
        visit::visit_expr(self, i);
    }
    fn visit_attribute(&mut self, i: &'ast syn::Attribute) {
        if i.path().is_ident("doc") {
            self.mark(i, EXCLUDE);
        }
    }
    fn visit_macro(&mut self, i: &'ast syn::Macro) {
        if i.path.is_ident("verus") {
            let file: syn::File = syn::parse2(i.tokens.clone()).expect("parse verus! body");
            self.visit_file(&file);
        }
    }
}

fn count(source: &str) -> serde_json::Value {
    let file = syn::parse_file(source).expect("parse Rust/Verus source");
    let mut c = Counter {
        lines: vec![IMPL; source.lines().count()],
        code: BTreeSet::new(),
        trusted: vec![],
    };
    c.tokens(source.parse().expect("tokenize Rust/Verus source"));
    c.visit_file(&file);
    let mut surface = Surface::default();
    surface.visit_file(&file);
    let source_lines: Vec<_> = source.lines().collect();
    let mut groups = [vec![], vec![], vec![], vec![]];
    for n in &c.code {
        if *n > 0 && *n <= c.lines.len() && !source_lines[*n - 1].trim().is_empty() {
            groups[c.lines[*n - 1] as usize].push(*n);
        }
    }
    serde_json::json!({"implementation": groups[0], "proof": groups[1],
                       "specification": groups[2], "excluded":groups[3], "trusted":c.trusted,
                       "surface":surface.units})
}

fn main() {
    let mut result = serde_json::Map::new();
    for path in env::args().skip(1) {
        let source = fs::read_to_string(&path).expect("read source");
        eprintln!("Counting {path}");
        result.insert(path, count(&source));
    }
    println!("{}", serde_json::Value::Object(result));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn surface_stops_at_proof_signature() {
        let r = count("verus! {\nproof fn top() requires meaning(), {\n helper();\n}\nopen spec fn meaning() -> bool { true }\n}");
        let units = r["surface"].as_array().unwrap();
        let root = units.iter().find(|u| u["name"] == "top").unwrap();
        assert_eq!(root["end"], 2);
    }
    #[test]
    fn surface_recognizes_uninterpreted_and_external_assumptions() {
        let r = count("verus! {\npub uninterp spec fn meaning() -> int;\n#[verifier::external_body]\nproof fn law() ensures true, {}\n}");
        assert!(r["surface"].as_array().unwrap().iter().all(|u| u["assumption"] == true));
    }
    #[test]
    fn multiline_assumption_records_complete_extent() {
        let r = count("verus! {\nproof fn p() {\n assume(\n true\n );\n}\n}");
        assert_eq!(r["trusted"][0]["line"], 3);
        assert_eq!(r["trusted"][0]["end"], 5);
    }
    #[test]
    fn contracts_inline_proof_and_ghost() {
        let s = "verus! {\nfn f()\n requires true,\n{\n let x = 1;\n let ghost y = 2;\n proof { assert(y == 2); }\n assert(x == 1);\n}\n}";
        let r = count(s);
        assert_eq!(r["specification"], serde_json::json!([3]));
        assert_eq!(r["proof"], serde_json::json!([6, 7, 8]));
    }
    #[test]
    fn spec_and_assumed_proofs_are_not_checked_proof_loc() {
        let s = "verus! {\n// comment\n#[verifier::external_body]\nproof fn imported() ensures true, {}\nopen spec fn value() -> int { 1 }\n}";
        let r = count(s);
        assert_eq!(r["proof"], serde_json::json!([]));
        assert_eq!(r["specification"], serde_json::json!([3, 4, 5]));
        assert_eq!(r["trusted"].as_array().unwrap().len(), 1);
    }
    #[test]
    fn loop_invariants_and_documentation() {
        let s = "verus! {\n/// Documentation\nfn f() {\n let mut i = 0;\n while i < 4\n invariant i <= 4,\n decreases 4 - i,\n { i = i + 1; }\n}\n}";
        let r = count(s);
        assert_eq!(r["specification"], serde_json::json!([6, 7]));
        assert!(r["excluded"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!(2)));
    }
    #[test]
    fn embedded_tests_and_disabled_items_are_excluded() {
        let r = count("#[cfg(test)]\nmod tests { fn check() {} }\nverus! {\n#[cfg(any())]\nproof fn unused() {}\n}");
        assert_eq!(r["proof"], serde_json::json!([]));
        assert_eq!(r["excluded"], serde_json::json!([1, 2, 4, 5]));
    }
    #[test]
    fn ghost_types_and_spec_constants() {
        let r = count("verus! {\npub spec const N: nat = 4;\npub ghost enum E { A, B }\n}");
        assert_eq!(r["specification"], serde_json::json!([2, 3]));
    }
}
