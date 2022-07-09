The goal here is to predict the argumentative effectiveness of each paragraph in the essay.

They have shared a dataset from previous competition, in which we had to predict, what the paragraph was talking about, Lead, evidence, etc. Use that data, and pre-training task to fine-tune the model for this type of dataset. It is already great at this, but should be able to give more focus to this type of data after fine-tuning like that. We will also, make a small sample test dataset whose essay we can exclude from the fine-tuning, if need be. But I don't think that should effect anything. But we will try with both. 

Ideas
1. Using discourse type as a causal input. 
2. Distilling for effectiveness
3. Checkout sparsely activated models for effectiveness too!
4. Explainable Transformers library to figure out where we are going wrong.
