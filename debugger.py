if __name__ == '__main__':

    from transformers import TapasConfig,TapasTokenizer,TapasForMaskedLM
    from transformers import pipeline
    import pandas as pd
    import numpy as np
    import torch
    import sys

    config = TapasConfig.from_pretrained(
        'google/tapas-base-finetuned-wtq',from_pt=True)
    model = TapasForMaskedLM.from_pretrained(
        'google/tapas-base-finetuned-wtq', config=config)
    tokenizer=TapasTokenizer.from_pretrained(
        "google/tapas-base-finetuned-wtq", from_pt=True)

    # outdir = "tmp"

    # model.save_pretrained(outdir)
    # tokenizer.save_pretrained(outdir)
    # config.save_pretrained(outdir)


    nlp = pipeline(task="fill-mask",framework="pt",model=model, tokenizer=tokenizer)
    #nlp = pipeline(task="table-question-answering")


    data= {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": ["56", "45", "59"],
        "number of movies": ["87", "53", "69"],
        "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"]
    }

    table = pd.DataFrame.from_dict(data)

    queries=[
        f"The number of movies Brad Pitt acted in is {tokenizer.mask_token}",
        f"Leonardo di caprio's age is {tokenizer.mask_token}"]

    test = nlp(queries, table = pd.DataFrame(table))