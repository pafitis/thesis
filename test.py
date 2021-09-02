from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

# Define the table
data = {'Cities': ["Paris, France", "London, England", "Lyon, France"], 'Inhabitants': ["2.161", "8.982", "0.513"]}

# Define the questions
queries = ["Which city has most inhabitants?", "What is the average number of inhabitants?", "How many French cities are in the list?", "How many inhabitants live in French cities?"]

def load_model_and_tokenizer():
  """
    Load
  """
  # Load pretrained tokenizer: TAPAS finetuned on WikiTable Questions
  tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")

  # Load pretrained model: TAPAS finetuned on WikiTable Questions
  model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

  # Return tokenizer and model
  return tokenizer, model


def prepare_inputs(data, queries, tokenizer):
  """
    Convert dictionary into data frame and tokenize inputs given queries.
  """
  # Prepare inputs
  table = pd.DataFrame.from_dict(data)
  inputs = tokenizer(table=table, queries=queries, padding='max_length', return_tensors="pt")
  
  # Return things
  return table, inputs


def generate_predictions(inputs, model, tokenizer):
  """
    Generate predictions for some tokenized input.
  """
  # Generate model results
  outputs = model(**inputs)

  # Convert logit outputs into predictions for table cells and aggregation operators
  predicted_table_cell_coords, predicted_aggregation_operators = tokenizer.convert_logits_to_predictions(
          inputs,
          outputs.logits.detach(),
          outputs.logits_aggregation.detach()
  )
  
  # Return values
  return predicted_table_cell_coords, predicted_aggregation_operators


def postprocess_predictions(predicted_aggregation_operators, predicted_table_cell_coords, table):
  """
    Compute the predicted operation and nicely structure the answers.
  """
  # Process predicted aggregation operators
  aggregation_operators = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3:"COUNT"}
  aggregation_predictions_string = [aggregation_operators[x] for x in predicted_aggregation_operators]

  # Process predicted table cell coordinates
  answers = []
  for coordinates in predicted_table_cell_coords:
    if len(coordinates) == 1:
      # 1 cell
      answers.append(table.iat[coordinates[0]])
    else:
      # > 1 cell
      cell_values = []
      for coordinate in coordinates:
        cell_values.append(table.iat[coordinate])
      answers.append(", ".join(cell_values))
      
  # Return values
  return aggregation_predictions_string, answers


def show_answers(queries, answers, aggregation_predictions_string):
  """
    Visualize the postprocessed answers.
  """
  for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
    print(query)
    if predicted_agg == "NONE":
      print("Predicted answer: " + answer)
    else:
      print("Predicted answer: " + predicted_agg + " > " + answer)


def run_tapas():
  """
    Invoke the TAPAS model.
  """
  tokenizer, model = load_model_and_tokenizer()
  table, inputs = prepare_inputs(data, queries, tokenizer)
  predicted_table_cell_coords, predicted_aggregation_operators = generate_predictions(inputs, model, tokenizer)
  aggregation_predictions_string, answers = postprocess_predictions(predicted_aggregation_operators, predicted_table_cell_coords, table)
  show_answers(queries, answers, aggregation_predictions_string)


if __name__ == '__main__':
  run_tapas()