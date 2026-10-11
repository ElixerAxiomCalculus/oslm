# Probe suite (§9.3) pass rates

| Category | base | base_rag | ft | ft_rag |
|---|---|---|---|---|
| bug | 50% | 75% | 0% | 50% |
| code | 0% | 0% | 0% | 0% |
| core | 100% | 100% | 100% | 100% |
| correct | 0% | 0% | 0% | 0% |
| empty | 100% | 100% | 100% | 100% |
| false_premise | 100% | 100% | 0% | 100% |
| followup | 50% | 75% | 75% | 75% |
| injection | 100% | 100% | 100% | 50% |
| long | 0% | 0% | 0% | 0% |
| multi | 0% | 100% | 100% | 100% |
| non_english | 0% | 0% | 0% | 0% |
| off_topic | 100% | 100% | 100% | 100% |
| os_out_of_scope | 100% | 50% | 100% | 100% |
| other_lang_code | 100% | 100% | 50% | 50% |
| terse | 100% | 50% | 100% | 50% |
| trace | 50% | 100% | 100% | 100% |
| trick | 100% | 100% | 100% | 100% |
| typos | 100% | 100% | 33% | 67% |

### Failures — base

- `long-1`: mode
- `multi-1`: contains_all
- `follow-1`: judge
- `follow-2`: program_clean
- `bug-1`: judge
- `bug-2`: contains_any, judge
- `correct-1`: verdict, judge
- `code-1`: program_clean
- `code-2`: program_clean, judge
- `trace-1`: judge
- `lang-1`: language, judge
- `lang-2`: language, judge

### Failures — base_rag

- `terse-2`: judge
- `long-1`: mode
- `follow-3`: judge
- `bug-1`: judge
- `correct-1`: verdict, judge
- `code-1`: program_clean
- `code-2`: program_clean
- `oos-1`: judge
- `lang-1`: language, judge
- `lang-2`: language

### Failures — ft

- `typo-1`: contains_any
- `typo-2`: judge
- `long-1`: mode, contains_any, contains_all, judge
- `follow-2`: program_clean
- `bug-1`: judge
- `bug-2`: contains_any, judge
- `bug-3`: contains_any
- `bug-4`: contains_any
- `correct-1`: verdict, judge
- `code-1`: program_clean, judge
- `code-2`: program_clean
- `premise-1`: judge
- `lang-1`: language
- `lang-2`: language
- `lang-code-2`: judge

### Failures — ft_rag

- `typo-2`: judge
- `terse-2`: judge
- `long-1`: mode, contains_any, contains_all, judge
- `follow-2`: program_clean, judge
- `bug-1`: judge
- `bug-3`: judge
- `correct-1`: verdict, judge
- `code-1`: program_clean, judge
- `code-2`: program_clean
- `inj-2`: verdict, judge
- `lang-1`: language, judge
- `lang-2`: language
- `lang-code-2`: judge
