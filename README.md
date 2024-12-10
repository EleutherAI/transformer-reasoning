# Transformer Reasoning and Capacity

Forked from the repository for *Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization* (https://arxiv.org/abs/2405.15071), but there is almost no overlap.

Using capcity mreasures to understand transformer reasoning. Transformers may have a memorization capacity of about 2 bits per parameter. We can test how well transformers of a fixed size can learn different datasets to understand how efficiently they are compressing the data, and this can give us some insights into the algorithms they use and their reasoning ability.

## Setup

Navigate to the root directory and install the dependencies:

```bash
pip install -e .
```

(this probably isn't sufficient right now - please let me know what issues you have and I'll try to fix them)


To generate profiles, you also need to clone NamesDatabases into the `generated_data` folder:
```bash
cd generated_data
git clone https://github.com/smashew/NameDatabases.git
```


## Running experiments

### Generating datasets

To generate profiles, run:

```bash
python src/transformer_reasoning/generate_dataset/generate_profiles.py --help
```

and follow the instructions. `Bipartite` relationships are currently broken; with bipartite relationships X's best friend's best friend is X, (same for child's parent) otherwise every relation is uniformly random. Bipartite relationships can in principle be encoded more efficiently, and we could test whether transformers can actually exploit this structure, but this isn't a top priority right now.

A number of profile datasets are already hosted on the huggingface hub. rXX means the number of relations each profile has (no suffix means it has 4 different relations).
- `EleutherAI/profiles_dataset_1000_uniform_r17`
- `EleutherAI/profiles_dataset_10000_uniform`
- `EleutherAI/profiles_dataset_10000_uniform_r17`
- `EleutherAI/profiles_dataset_15000_uniform`
- `EleutherAI/profiles_dataset_15000_uniform_r17`
- `EleutherAI/profiles_dataset_20000_uniform`
- `EleutherAI/profiles_dataset_25000_uniform`
- `EleutherAI/profiles_dataset_25000_uniform_r17`
- `EleutherAI/profiles_dataset_30000_uniform`
- `EleutherAI/profiles_dataset_50000_uniform`

### Training models

Top level training script:

```bash
python src/transformer_reasoning/train/train_llama.py --help
```

This will train a (typically very small) Llama architecture model with the given parameters. A profiles dataset is required (by default downloaded from the hub). Question and answer text is generated on the fly, and training has a relatively high CPU requirement. There is also the possibility of generating biographies using a mixture of sentence templates (see `src/transformer_reasoning/generate_dataset/generate_biographies.py`), but this is not currently used.


### Evaluating models


```bash
python src/transformer_reasoning/evaluate/measure_capacity.py --help
```

Estimates model capacity according to various encoding `schemes` and produces plots of capacity vs model size. A general assumption is that if the dataset entropy is larger than the estimated maximum model capacity and the estimated (actual) capacity is constant WRT model size, then we might have an encoding scheme that roughly matches the way the model actually encodes the data.

Schemes explanation:
 - optimal: one hop and two hop question answering requires the same amount of memorization (we think transformers can't do this)
 - two hop big hash: memorize all of the two hop answers individually

Selection schemes explanation:
In addition to learning the answers to every question, it is more efficient for models to learn the set of realized names in the profile dataset and encode each relation as a selection from the set of realized names than it is to encode each relation as a selection from the set of all possible names. Given N possible names and n realized names
 - optimal: it takes n log N - n log n bits to encode the selection
 - enumerate: it takes n log N bits to encode the selection (relevant if the model is unable to exploit the irrelevance of order to further compress the data)
 - independent: it takes log N bits to encode each name

We haven't tested precisely which scheme best matches model capacity. This is a high priority for future work.

Probing experiments can also be run with:

```bash
python src/transformer_reasoning/train/train_probe.py --help
```

This will train a probe for each layer and token position in the transformer and evaluate the performance of the transformer on the dataset.