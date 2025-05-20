# LLM experimental code for Adaptive Learn-then-Test: Statistically Valid and Efficient Hyperparameter Selection

This code is built based on https://github.com/keirp/automatic_prompt_engineer 

- aLTT is implemented at /automatic_prompt_engineer/altogether_ape.py

- Code to run instruction induction experiment can be found at /experiments/main.py

- Example execution of the code for the target risk alpha and tolerance delta reads:
 ```
python main.py --risk_control_mode ['FWER' or 'FDR'] --alpha $alpha$ --delta $delta$
```

- Generated prompts using Llama3.3 70B (entire candidate set) as well as corresponding loss table obtained from Llama3 8B Instruct can be found in experiments/cache_original while 20 different random split of validation and testing data can be found in experiments/cache
- Saved results with plot-generating code can be found in /plotting_with_saved_results/