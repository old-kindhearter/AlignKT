# AlignKT: Explicitly Modeling Knowledge State for Knowledge Tracing with Ideal State Alignment

This is the code for the paper *AlignKT: Explicitly Modeling Knowledge State for Knowledge Tracing with Ideal State Alignment*. Our paper see here(The link for the paper will be updated here).

![model_framework](A:\pStudy\AlignKT\model_framework.png)

## Requirements

```
torch>=2.1.0
```

## Arguments

- `--dataset_name:` `default value=assist2009, value={algebra2005, nips_task34}`
- `--d_model:` `default value=384(for AS09), value={256(for AL05, NIPS34)}`
- `--d_ff:` `default value=1024(for AS09, NIPS34), value={768(for AL05)}`
- `--n_heads:` `default value=4(for AS09), value={8(for AL05, NIPS34)}`
- `--drop_out:` `default value=0.25(for AS09), value={0.2(for AL05, NIPS34)}`
- `--batch_size:`  `default value=128(for AS09), value={32(for AL05, NIPS34)}`

## Run

`python main.py --seed 42`

## Citation

```
The citation information for the paper will be updated here.
```

## Acknowledgements

The code for model training and real-scenario evaluation is sourced from [pykt-team/pykt-toolkit: pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models](https://github.com/pykt-team/pykt-toolkit)
