import logging

import azure.functions as func

from square_skill_api.models.prediction import QueryOutput, QueryRequest

from square_skill_helpers.square_api import ModelAPI

logger = logging.getLogger(__name__)

model_api = ModelAPI()


async def main(req: func.HttpRequest) -> func.HttpResponse:

    request_body = req.get_json()

    query = request_body["query"]
    context = request_body["skill_args"]["context"]
    prepared_input = [context, query] 
    
    model_request = { 
        "input": prepared_input,
        "preprocessing_kwargs": {},
        "model_kwargs": {},
        "adapter_name": "AdapterHub/bert-base-uncased-pf-boolq"
    }
    model_api_output = await model_api(
        model_name="bert-base-uncased", 
        pipeline="sequence-classification", 
        model_request=model_request
    )
    logger.info(f"Model API output:\n{model_api_output}")

    response = QueryOutput.from_sequence_classification(
        answers=["no", "yes"], 
        model_api_output=model_api_output, 
        context=context
    ).json()
    
    return func.HttpResponse(response, status_code=200)

