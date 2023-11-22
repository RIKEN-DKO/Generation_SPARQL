# KQA Pro version 1.0

KQA Pro is a large-scale dataset of complex question answering over knowledge base. The questions are very diverse and challenging, requiring multiple reasoning capabilities including compositional reasoning, multi-hop reasoning, quantitative comparison, set operations, and etc. Strong supervisions of SPARQL and program are provided for each question.
If you find our dataset is helpful in your work, please cite us by

```
@inproceedings{KQAPro,
  title={{KQA P}ro: A Large Diagnostic Dataset for Complex Question Answering over Knowledge Base},
  author={Cao, Shulin and Shi, Jiaxin and Pan, Liangming and Nie, Lunyiu and Xiang, Yutong and Hou, Lei and Li, Juanzi and He, Bin and Zhang, Hanwang},
  booktitle={ACL'22},
  year={2022}
}
```

## Usage
There are four json files included in our dataset:

- `kb.json`, the target knowledge base used to answer questions, which is a dense subset of [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page).
- `train.json`, the training set, including 94,376 QA pairs with annotations of SPARQL and program for each.
- `val.json`, the validation set, including 11,797 QA pairs with SPARQL and program.
- `test.json`, the test set, including 11,797 questions, with 10 candidate choices for each. You can submit your predictions and your performance will be shown in our leaderboard.

Following is the detailed formats

**kb.json**
```
{
    'concepts':
    {
        '<id>':
        {
            'name': str,
            'instanceOf': ['<id>', '<id>'], # ids of parent concept
        }
    },
    'entities': # excluding concepts
    {
        '<id>': 
        {
            'name': str,
            'instanceOf': ['<id>', '<id>'], # ids of parent concept
            'attributes':
            [
                {
                    'key': str, # attribute key
                    'value':  # attribute value
                    {
                        'type': 'string'/'quantity'/'date'/'year',
                        'value': float/int/str, # float or int for quantity, int for year, 'yyyy/mm/dd' for date
                        'unit': str,  # for quantity
                    },
                    'qualifiers':
                    {
                        '<qk>':  # qualifier key, one key may have multiple corresponding qualifier values
                        [
                            {
                                'type': 'string'/'quantity'/'date'/'year',
                                'value': float/int/str,
                                'unit': str,
                            }, # the format of qualifier value is similar to attribute value
                        ]
                    }
                },
            ]
            'relations':
            [
                {
                    'predicate': str,
                    'object': '<id>', # NOTE: it may be a concept id
                    'direction': 'forward'/'backward',
                    'qualifiers':
                    {
                        '<qk>':  # qualifier key, one key may have multiple corresponding qualifier values
                        [
                            {
                                'type': 'string'/'quantity'/'date'/'year',
                                'value': float/int/str,
                                'unit': str,
                            }, # the format of qualifier value is similar to attribute value
                        ]
                    }
                },
            ]
        }
    }
}
```

**train.json/val.json**
```
[
    {
        'question': str,
        'sparql': str, # executable in our virtuoso engine
        'program': 
        [
            {
                'function': str,  # function name
                'dependencies': [int],  # functional inputs, representing indices of the preceding functions
                'inputs': [str],  # textual inputs
            }
        ],
        'choices': [str],  # 10 answer choices
        'answer': str,  # golden answer
    }
]
```

**test.json**
```
[
    {
        'question': str,
        'choices': [str],  # 10 answer choices
    }
]
```

## How to run SPARQLs and programs
We implement multiple baselines in our [codebase](https://github.com/shijx12/KQAPro_Baselines), which includes a supervised SPARQL parser and program parser.

In the SPARQL parser, we implement a query engine based on [Virtuoso](https://github.com/openlink/virtuoso-opensource.git).
You can install the engine based on our [instructions](https://github.com/shijx12/KQAPro_Baselines/blob/master/SPARQL/README.md), and then feed your predicted SPARQL to get the answer.

In the program parser, we implement a rule-based program executor, which receives a predicted program and returns the answer.
Detailed introductions of our functions can be found in our [paper](https://arxiv.org/abs/2007.03875).

## How to submit results of test set
You need to predict answers for all questions of test set and write them in a text file **in order**, one per line.
Here is an example:
```
Tron: Legacy
Palm Beach County
1937-03-01
The Queen
...
```

Then you need to send the prediction file to us by email <caosl19@mails.tsinghua.edu.cn>, we will reply to you with the performance as soon as possible.
To appear in the learderboard, you need to also provide following information:

- model name
- affiliation
- open-ended or multiple-choice
- whether use the supervision of SPARQL in your model or not
- whether use the supervision of program in your model or not
- single model or ensemble model
- (optional) paper link
- (optional) code link


## Contact
If you have any questions, feel free to contact <shijx12@gmail.com>.
