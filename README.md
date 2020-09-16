# plan-bert-aaai
PLAN-BERT is the course recommendation system introduced in paper "Degree Planning with PLAN-BERT: Multi-Semester Recommendation UsingFuture Courses of Interest".

| Required package | Version |
| ---- | --- |
| Python | 3.6.3 |
| Keras | 2.3.0 |
| Numpy | 1.18.2 |
| pynvml | 8.0.3 |
| tqdm | 4.15.0 |


| Component | Description |
| ---- | --- |
| **UNIVERSITY1** | The code of variants of PLAN-BERT and baselines in UNIVERSITY1. |
| **UNIVERSITY1/PLAN-BERT+item+user.py** | The code of the training procedure of the optimal PLAN-BERT in UNIVERSITY1. |
| **SYSTEM1** | The code of variants of PLAN-BERT and baselines in SYSTEM1. |
| **SYSTEM1/PLAN-BERT+item+user.py** | The code of the training procedure of the optimal PLAN-BERT in SYSTEM1. |
| **model** | The definition of the architectures of PLAN-BERT and baselines. We heavily employ the code of https://github.com/kpot/keras-transformer. |
| **model/PLANBERT.py** | The definition of the architectures of PLAN-BERT. |
| **util** | The code Generators, Loss Functions, and Dataloaders. |

