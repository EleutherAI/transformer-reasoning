## Objectives

 - Get (predictable?) learning of 2-hop QA
 - Scaling curves for 2 hop learning
  - depth scaling
 - Get very low loss on 2-hop QA
 - What effect does pretraining have?
 - How does 2-hop learning chance with function complexity?


## Ideas

 - 2 hop only QA data, heavy QA weight in data mix
 - QA masked loss
 - zero WD

## Rules of the thumb

 - 25k profiles ~ saturate 1M params
 - 10k profiles ~ saturate 400k params
 - 3M steps (?) for 2hop convergence on infinite dataset near saturation
 - 

## Schedule of experiments:

 - learning 2-hop QA + scaling curves:
   - 