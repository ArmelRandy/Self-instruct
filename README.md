# Self-instruct 🤗
A repository to perform self-instruct with a model on Hugging Face Hub.

# What is this about?
This repository is dedicated to [Self-instruct](https://arxiv.org/pdf/2212.10560.pdf). It is an iterative approach which allows to generate a dataset of instructions by boostrapping on a model's prediction. For it to work well, the model used has to be powerful. The original work actually focuses on OpenAI's `text-davinci-003` engine which is one of their most powerful model. Our aim is to give a chance to modest, decoder-based models to be used for a data generation purpose.

# News

* **September 6, 2023**: Get ready to welcome self-instruct data from [Code Llama](https://huggingface.co/codellama).
* **May 24, 2023:** We've built a space which allow to visualize the data generated by self-instruct when the model used is [StarCoder💫](https://arxiv.org/pdf/2305.06161.pdf), the recent SOTA open-source code LLM by Hugging Face 🤗.

# Disclaimer

- Our approach requires the availability of a good amount of computational resources/ an inference endpoint.
- We will focus on the dataset generation pipeline and the curation rather than the fine-tuning.
- Keep in mind that the quality of the dataset obtained by this method is strongly dependent on the quality of the model that is used. 

# Table of Contents
1. [Overview of the method](#overview)
2. [Related work](#related-work)
3. [Our approach](#our-approach)
    - [The prompting format](#the-prompting-format)
    - [The trigger words](#the-trigger-words)
    - [The post-processing](#the-post-processing)
        - [Self-consistency](#self-consistency)
        - [Uniqueness](#uniqueness)
5. [Quickstart](#quickstart)
    - [Step by step installation with conda](#step-by-step-installation-with-conda)
    - [Instruction-output](#instruction-output)
    - [Instruction-input-output](#instruction-input-output)
    - [Text-generation-inference support](#text-generation-inference-support)
    - [Post-processing](#post-processing)
        - [Self-consistency](#self-consistency)
        - [Uniqueness](#uniqueness)
    - [Visualization and statistics](#visualization-and-statistics)
6. [Fine-tuning](#fine-tuning)
7. [Acknowledgements](#acknowledgments)

# Overview
Self-instruct is an iterative method that helps LM improve their ability to follow natural language instructions. The idea is to use a seed set of manually-written instructions and use them to prompt the model to generate new instructions and their corresponding input-output instances. The method includes a filtering step to ensure the novelty of the generated task.

# Related work
Our implementation is inspired by the original [Self-instruct](https://github.com/yizhongw/self-instruct) method and recent updates including [Stanford's alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/) and [Code alpaca](https://github.com/sahil280114/codealpaca/). While the last two are almost identical, with the sole difference being the set of seed tasks used, the original work has a different mindset. As a matter of fact, self-instruct's author uses a set of seed tasks and prompt the model with some of them to make it generate instructions. Later on, the output to the generated instructions are found separately. Conversely, Alpaca is all in one in the sense that the model is prompted to generate an instruction as well as the input-output pair at the same time. It uses the following template

```bash
### Instruction:
{instruction}

### Input:
{input}

### Output:
{output}
```
The advantage is that this all in one template allows to reduce the inference cost of the method, and the quality of the generated instances is not proven to be significantly impaired. We believe, intuitively, that this prompting approach generates feasible instructions thanks to the obligation to have a sound input-output pair associated to it.

# Our approach

Our approach is focused on code use cases, therefore our modifications are mostly relevant for that framework.

## The prompting format
During our tests, we realized that, at least with "small" code models, the trigger words `Input` and `Output` tend to make them generate test cases instead. It is significantly impairing because given an instruction, we want a working implementation rather than a potentially buggy test case. In order to alleviate this issue, we decided to get rid of the `Input` trigger word. We adopt an instruction-output format.

## The trigger words
Using `Instruction`, `Input` and `Output` seems to work well for `text-davinci-003` but how well does it work for other models? This parameter is definitely relevant for small models as this can have a huge impact on the quality of their generations. Following this intuition, we included in our code the possibility to change the trigger words that are used during the prompting. This allows to accomodate to every single model.

## The post-processing
How to select and post-process the instructions that are generated by prompting a model? In the original work, the instructions are generated iteratively, and we keep those with a rouge score stricly less than `0.7` with any previously generated instruction. This allows diversity in the dataset, at least in terms of how the instructions are worded. According to our experiments, it is still possible to generate a problem multiple times with a different formulation each time. We propose to extend take the curation further with multiple ideas.

### Self-consistency
We came up with a strong data instruction filtering technique. The idea is very simple, we want to test if the model is consistent with what it generates. We verify that by prompting the model to generate and instruction based the output. It is a complicated task for a LM and for a human because in many cases, it results in an unsolvable task. In the case where the model is able to generate an instruction, we compare it in terms of meaning with the ground-truth. For that, we use [Sentence-BERT](https://arxiv.org/pdf/1908.10084.pdf), precisely [All-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) with the threshold of our choice (typically 0.5). This filtering technique is not recommended for models with a frailty ability to understand natural language text.

### Uniqueness
Another alternative is to post-process the raw dataset by only keeping instructions that are not similar to each other in terms of meaning. Once again we make use of Sentence-BERT. An instruction is kept if any previously generated instruction has a similarity score less than a threshold (typically 0.5) w.r.t the considered instruction.

## Further details
We modified the seed tasks to keep only those who are related to code. For that we combine the tasks from Code Alpaca (code tasks extrated from the original [seed tasks](https://github.com/yizhongw/self-instruct/blob/main/data/seed_tasks.jsonl) + some new tasks probably created by the repo's author) and some leetcode tasks. We have a total of `41` seed tasks.

# Quickstart

StarCoder was trained on GitHub code, thus it can be used to perform code generation. More precisely, the model can complete the implementation of a function or infer the following characters in a line of code. This can be done with the help of the 🤗's [transformers](https://github.com/huggingface/transformers) library.

## Step by step installation with conda

Here, we present a step by step recipe that anybody can use in order to apply our self-instruct method on its prefered LLM in a conda environment.
Create a new conda environment and activate it
```bash
conda create -n env
conda activate env
```
Install the `pytorch` version compatible with your version of cuda [here](https://pytorch.org/get-started/previous-versions/), for example the following command works with cuda 11.6
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
Install `transformers` and `accelerate`
```bash
conda install -c huggingface transformers 
pip install git+https://github.com/huggingface/accelerate
```
Do not forget to launch `accelerate config` in the terminal in order to configure you environment, for more the details see [accelerate](https://github.com/huggingface/accelerate).

We will also need [`rouge-score`](https://github.com/google-research/google-research/tree/master/rouge)

```bash
pip install rouge-score
```
Now we are ready to clone the repository and to start working 
```bash
git clone https://github.com/ArmelRandy/self-instruct
cd Self-instruct
```

## Instruction - output

Here we prompt the model with the following template
```bash
### Instruction :
{instruction}

### Output :
{output}
```
For the instructions that provides an input (a code in case of a debugging task or a translation task), we concatenate the instruction and the input under the keyword `Instruction`, we then have 
```bash
### Instruction :
{instruction}
{input}

### Output :
{output}
```
The possibility to change the trigger words `Instruction` and `Output` into other words such as `Request` and `Answer` respectively for example is given. However, the change has to be done directly in the code, you'll need to define a template and add it in `template.py`.

```bash
accelerate launch main.py \
    --seed_tasks_path="data/code_tasks.jsonl" \
    --output_data_path data/output.jsonl \
    --num_instructions_to_generate 20000 \
    --template_name better \
    --format 2 \
    --model_name_or_path="bigcode/starcoderbase-1b" \
    --num_prompt_instructions 8 \
    --request_batch_size 8 \
    --num_prompt_synthetic_instructions 2 \
    --max_new_tokens 4096 \
    --temperature 0.8 \
    --top_p 0.95 \
    --num_beams 1 \
    --repetition_penalty 1.2 \
    --threshold 0.7 \
    --seed 42 \
    --keep_programming \
```

## Instruction - input - output

It is the template as designed in Stanford's alpaca. The possibilty to change the trigger words is also provided, with the same limitations as those previously mentionned.

```bash
accelerate launch main.py \
    --seed_tasks_path="data/code_tasks.jsonl" \
    --output_data_path data/output.jsonl \
    --num_instructions_to_generate 20000 \
    --template_name better \
    --format 3 \
    --model_name_or_path="bigcode/starcoderbase-1b" \
    --num_prompt_instructions 8 \
    --request_batch_size 8 \
    --num_prompt_synthetic_instructions 2 \
    --max_new_tokens 4096 \
    --temperature 0.8 \
    --top_p 0.95 \
    --num_beams 1 \
    --repetition_penalty 1.2 \
    --threshold 0.7 \
    --seed 42 \
    --keep_programming \
```
## Text-generation-inference support
It is possible to use [TGI](https://github.com/huggingface/text-generation-inference) for the data generation if you have access to an inference endpoint. You'll need to set your hugging face token and the url of your endpoint in the environment variables `HF_TOKEN` and `API_URL`. In order to use TGI, you'll need to add `--use_tgi` to the above commands.


## Post-processing 
This part requires an additional requirement, that is [sentence-transformers](https://www.sbert.net) whose installation is as follows :

```bash
pip install -U sentence-transformers
```
### Self-consistency

Here, we run the file `processing.py` with the help of `accelerate`

```bash
accelerate launch processing.py \
    --seed_tasks_path="data/code_tasks.jsonl" \
    --input_data_path data/output.jsonl \
    --output_data_path data/output_processed.jsonl \
    --template_name default \
    --model_name_or_path="bigcode/starcoderbase-1b" \
    --num_prompt_instructions 4 \
    --num_trials 1 \
    --max_new_tokens 512 \
    --temperature 0.2 \
    --top_p 0.95 \
    --num_beams 1 \
    --repetition_penalty 1.2 \
    --threshold 0.7 \
    --seed 42 \
```

### Uniqueness

Here we want to apply a post-processing to our generated instructions by considering only instructions that are not too similar. In order to do so, we get into the folder `self-instruct` and we launch

```bash
cd post_processing
python unique_post_processing.py \
    --input_data_path ../data/output.jsonl \
    --output_data_path ../data/output_processed.jsonl \
    --threshold 0.5 \
```

## Visualization and statistics
It is possible to visualize the instructions generated in terms of how they are phrased. Specifically we can show the most common used root verbs and their top 4 direct noun objects. This functionality is inherited from the [implementation](https://github.com/yizhongw/self-instruct/blob/main/self_instruct/instruction_visualize.ipynb) provided by self-instruct's author. Its usage requires additional libraries, [spacy](https://spacy.io/usage), [benepar](https://github.com/nikitakit/self-attentive-parser) and [plotly](https://plotly.com/python/)
```bash
pip install -U spacy
python -m spacy download en_core_web_md
pip install benepar 
pip install plotly
```
Now, it is possible to run the notebook `instruction_visualize.ipynb`. We also provide `dataset_to_hub.ipynb` in order to push the generated dataset to the hub.

# Fine-Tuning

Now that the dataset is available, we can fine-tune our favorite text/code LLM to make it follow instructions. Our choice is naturally towards StarCoder. This [repository](https://github.com/bigcode-project/starcoder) gives a comprehensive method that can be used to fine-tune starcoder on any instruction dataset available on the hub. You can also check out [Octopack's repository](https://github.com/bigcode-project/octopack#citation).

# Acknowledgements

- [The original self-instruct method](https://github.com/yizhongw/self-instruct)
- [Stanford's Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Code Alpaca](https://github.com/sahil280114/codealpaca)
