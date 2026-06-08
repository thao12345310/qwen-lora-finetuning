# Dataset mix report: v4

## Split totals
- train: 20508
- valid: 2000
- test: 1997

## Source counts
- train: base_mimo=15998, patch_b2b3=790, patch_b4=1700, patch_short_contextual=2020
- valid: base_mimo=2000
- test: base_mimo=1997

## Train distribution
- by_domain: calling=2279, charging=2870, climate=1797, driver_assist=2004, messaging=2495, music=2471, navigation=2424, smart_home=1623, vehicle=2545
- by_context_required: false=5680, true=14828
- by_pattern: ack_after_done=220, ack_bind_proposal=240, ack_no_action=300, c1_language_control=200, confirm_with_modifier=160, g1_no_pronoun_content=150, g2_separate_time=120, g3_email_intent=120, g4_keep_location_device=200, g5_no_carry_prev_action=200, list_choice=140, n1_keep_negative_clause=250, n2_anti_inversion=150, n3_exclusion_in_list=250, n4_confirm_proposed=150, p1_pronoun_resolution=300, pronoun_recent_entity=140, reject_subproposal=220, s1_keep_qualifier=250, s2_compound_two_actions=150, short_correction=200, unknown=15998, vague_execute=160, vague_execute_no_proposal=240

## Source guard summary
- base_mimo: loaded=20000, selected=20000, accepted=19995, dup_drop=0, bench_drop=5, shortfall=0
- patch_b2b3: loaded=790, selected=790, accepted=790, dup_drop=0, bench_drop=0, shortfall=0
- patch_b4: loaded=1700, selected=1700, accepted=1700, dup_drop=0, bench_drop=0, shortfall=0
- patch_short_contextual: loaded=2020, selected=2020, accepted=2020, dup_drop=0, bench_drop=0, shortfall=0
