from cvqa.model_dev.programs import ProgramSpec

# 36-39, 44-47 --> Exists ( F2_attr(all, F1(gray); z=size) )
# 40-43, 48,50,51 --> Count ( F2_attr(all, F1(gray); z=size) )
# 72 --> Count(F2(all, F1(red), z=to_the_right))
# 84 --> Count(F1)
# 85 --> Exists(F1)
# 86 --> A(F1, seed=shape)
# 87 --> A(F1, seed=material)
# 88 --> A(F1, seed=ball)
# 89 --> A(F1, seed=size)

q_types = {
    'f1': [85, 86, 87, 88, 89],
    'f1_count': [84],
    'f2a': [36, 37, 38, 39, 44, 45, 46, 47],
    'f2a_count': [40, 41, 42, 43, 48, 50, 51],
    'f2_count': [72]
}

program_spec = ProgramSpec({
    'A': True,
    'F': True,
    'F2A': True,
    'F2': True
})

program_types = {
    'f1': 'A ( F )',
    'f1_count': 'A ( F )',
    'f2a': 'A ( F2A ( F ) )',
    'f2a_count': 'A ( F2A ( F ) )',
    'f2_count': 'A ( F2 ( F ) )'
}


def filter_samples(ds, types):
    qfs = []
    for t in types:
        qfs += q_types[t]

    ds.samples = [s for s in ds.samples if s['question_family_index'] in qfs]


programs_mapping = {}
for k, families in q_types.items():
    prog = program_types[k]
    prog_tokens = program_spec.vocab.encode(prog.split(' '))
    for f in families:
        programs_mapping[f] = (prog, prog_tokens)

