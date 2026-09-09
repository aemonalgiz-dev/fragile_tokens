Every glitch-token detector I know of tests a token the same way: alone in the prompt, "repeat this." I spent the last several weeks measuring what that test misses.

I placed tokens into banks of random, ordinary contexts and scored how well eight open models, from 1.7B to 235B parameters, copied them in place. Between 0.4% and 10.9% of the tokens that pass the standard single-token probe fail inside ordinary text. The English word "according" copies alone at probability 0.999 and is deleted in 44 of 48 contexts.

Three things stood out:

1. The failure belongs to the token, not the context. A token's fragility on half the contexts predicts it on the other half at 0.82 to 0.95.

2. The failures are confident. The model deletes, substitutes, truncates, or translates the token at near-zero entropy, then finishes the task on the substituted content. Given a glitch token in an agentic task, it searched for the sentence with the token removed, retrieved nothing, and delivered a recommendation anyway.

3. Fragile tokens sit next to glitch tokens in embedding space on every model with untied embeddings. Every published detector calls them healthy.

If you build systems that assemble prompts programmatically, the token to worry about is not the one that fails alone. It is the one that passes alone and fails in the sentence you happened to build.

Full paper, with every prompt, table, and failing case: https://medium.com/p/0270e70c72a6

#LLM #NLP #MachineLearning #AIResearch
