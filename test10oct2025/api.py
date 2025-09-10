import uuid
import datetime
from typing import Literal

import httpx
import fastapi
from fastapi import responses, Request
from opentelemetry import trace

from app import auth
from app import config
from app import guardrail
from app import rate_limiter
from app import schemas
from app import translator
from app.templates import prepare
from app.templates import load_products
from app.db.feedback.interface import DbFeedbackInterface
from app.services import validation
from app.metrics import (
    faithfulness_gauge,
    answer_relevance_gauge,
    context_precision_gauge,
    similarity_gauge,
)
from app.config import FAITHFULNESS_THRESHOLD, RELEVANCE_THRESHOLD
import logging
from app.logging.logging_qa import log_low_quality_response

router = fastapi.APIRouter()


@router.get("/docs/openapi.yaml", include_in_schema=False)
def get_openapi_yaml() -> responses.FileResponse:
    """Get the OpenAPI YAML specification.

    Returns:
        responses.FileResponse: The OpenAPI YAML file.
    """
    return responses.FileResponse(path="public/openapi.yaml", media_type="text/yaml")


@router.get(
    "/chats", response_model=list[schemas.Chat], response_model_exclude_none=True
)
async def get_chats(
    limit: int = 10,
    offset: int = 0,
    sort: Literal["asc", "desc"] = "desc",
    fields: str = "id,createdAt",
    user=fastapi.Depends(auth.require_employee_role),
):
    """
    Endpoint to retrieve chat history for the authenticated user.
    """
    responses_requested = False

    user_obj = DbFeedbackInterface.insert_user(email=user["email"])
    if user_obj is None:
        user_obj = DbFeedbackInterface.get_user_by_email(email=user["email"])

    field_list = fields.split(",")
    if "id" not in field_list:
        raise fastapi.HTTPException(
            status_code=422,
            detail="Invalid fields parameter. 'id' is required.",
        )

    if "responses" in field_list:
        responses_requested = True

    db_chats = DbFeedbackInterface.get_chats_by_user(
        meta_user_id=user_obj.meta_user_id,
        limit=limit,
        offset=offset,
        order_by=sort,
        responses_requested=responses_requested,
    )

    chats = [schemas.Chat.model_validate(chat) for chat in db_chats]

    for chat in chats:
        for attr in schemas.Chat.model_fields.keys():
            if attr not in field_list:
                setattr(chat, attr, None)

    return chats


@router.post(
    "/chats",
    response_model=schemas.Chat,
    status_code=201,
)
async def create_chat(
    user=fastapi.Depends(auth.require_employee_role),
):
    """
    Endpoint to create a new chat for the authenticated user.
    """
    user_obj = DbFeedbackInterface.insert_user(email=user["email"])
    if user_obj is None:
        user_obj = DbFeedbackInterface.get_user_by_email(email=user["email"])

    chat_obj = DbFeedbackInterface.insert_chat(
        meta_user_id=user_obj.meta_user_id,
    )

    chat = schemas.Chat.model_validate(chat_obj)
    return chat


@router.get("/chats/{chatId}/responses", response_model=list[schemas.ChatResponse])
async def get_chat_responses(
    chatId: str,  # pylint: disable=invalid-name
    limit: int = 10,
    offset: int = 0,
    sort: Literal["asc", "desc"] = "asc",
    user=fastapi.Depends(auth.require_employee_role),
):
    """
    Endpoint to retrieve a specific chat by its ID for the authenticated user.
    """
    user_obj = DbFeedbackInterface.insert_user(email=user["email"])
    if user_obj is None:
        user_obj = DbFeedbackInterface.get_user_by_email(email=user["email"])

    validation.validate_chat_existence_access(
        chat_id=chatId, meta_user_id=user_obj.meta_user_id
    )

    db_responses = DbFeedbackInterface.get_responses_by_chat_id(
        chat_id=chatId, limit=limit, offset=offset, order_by=sort
    )

    responses_list = [
        schemas.ChatResponse.model_validate(response) for response in db_responses
    ]

    return responses_list


@router.post(
    "/chats/{chatId}/responses", response_model=schemas.ChatResponse, status_code=201
)
async def create_chat_response(
    chatId: str,
    request: schemas.ChatMessage,
    fastapi_request: Request,
    user=fastapi.Depends(auth.require_employee_role),
    products=fastapi.Depends(load_products.get_products),
) -> schemas.ChatResponse:
    """Endpoint to create a new response for a chat.

    Args:
        chatId (str): The ID of the chat to which the response belongs.
        user (dict, optional): The authenticated user.
        products (tuple, optional): A tuple containing unique and all products.

    Returns:
        schemas.ChatResponse: The created chat response.
    """
    tracer = trace.get_tracer("assistant")
    session_id = fastapi_request.headers.get("X-Session-ID", "")
    with tracer.start_as_current_span(
        "responses-handler",
        attributes={
            "chatId": chatId,
            "session_id": session_id,
            "user_id": getattr(user, "id", None),
            "endpoint": "/chats/{chatId}/responses",
        },
    ) as span:
        user_obj = DbFeedbackInterface.insert_user(email=user["email"])
        if user_obj is None:
            user_obj = DbFeedbackInterface.get_user_by_email(email=user["email"])

        validation.validate_chat_existence_access(
            chat_id=chatId, meta_user_id=user_obj.meta_user_id
        )

        rate_limiter.check_rate_limit(user["email"], fastapi_request)

        unique_products, all_products = products

        translation = await translator.translate(request.content, "en")
        source_lang = translation.get("source_lang")
        if source_lang == "en":
            query = request.content
        elif source_lang in ("de", "fr"):
            query = translation.get("translated_text")
        else:
            chat_response = schemas.ChatResponse(
                id="00000000-0000-0000-0000-000000000000",
                createdAt=datetime.datetime.now(),
                input=request.content,
                output=(
                    "Language not supported. Supported languages are English, "
                    "German and French"
                ),
                references=[],
            )
            return chat_response

        # Make the usage of guardrails optional (from main branch)
        if getattr(config.settings, "guardrail_enabled", True):
            guardrail_result = guardrail.guardrails(query)
            if not guardrail_result["relevant"]:
                chat_response = schemas.ChatResponse(
                    id="00000000-0000-0000-0000-000000000000",
                    createdAt=datetime.datetime.now(),
                    input=request.content,
                    output=guardrail_result["message"],
                    references=[],
                )
                return chat_response

        query = prepare.request_prepare(query, all_products)

        temp_response_id = str(uuid.uuid4())

        conversation_history = []
        old_responses = DbFeedbackInterface.get_responses_by_chat_id(
            chat_id=chatId,
            limit=config.settings.max_conversation_turns,
            offset=0,
            order_by="desc",
        )

        for response in old_responses[::-1]:  # Reverse to maintain chronological order
            response_obj = schemas.ChatResponse.model_validate(response)
            conversation_element = {"role": "user", "content": response.query}
            conversation_history.append(conversation_element)
            conversation_element = {"role": "assistant", "content": response.output}
            conversation_history.append(conversation_element)

        path = "/search/local"
        search_local_url = config.settings.knowledge_graph_api_url + path
        async with httpx.AsyncClient() as client:
            post_data = {
                "message": {"role": "assistant", "content": query},
                "stream": request.stream,
                "temperature": 0.5,
                "max_tokens": 200,
                "conversation_history": conversation_history,
            }
            response = await client.post(search_local_url, json=post_data, timeout=30.0)
            response.raise_for_status()

            knowledge_graph_response = response.json()

            chat_response = prepare.response_prepare(
                user_input=request.content,
                response=knowledge_graph_response,
                unique_products=unique_products,
                response_id=temp_response_id,
            )

            if source_lang != "en":
                translation_to_original = await translator.translate(
                    chat_response.output, source_lang
                )
                chat_response.output = translation_to_original.get(
                    "translated_text", chat_response.output
                )

            chat_response.output = prepare.format_references(chat_response.output)

            response_obj = DbFeedbackInterface.insert_response(
                chat_id=chatId,
                user_input=chat_response.input,
                query=query,
                output=chat_response.output,
                original_response=knowledge_graph_response,
                rating=None,
            )

            chat_response.id = response_obj.id
            chat_response.createdAt = response_obj.createdAt

            for idx, reference in enumerate(chat_response.references):
                DbFeedbackInterface.insert_response_reference(
                    reference_id=reference.id,
                    reference_type=reference.type,
                    idx=idx,
                    response_id=response_obj.id,
                )

        # RAGAS scoring (pseudo-code, can replace with actual RAGAS usage)
        ragas_scores = evaluate_with_ragas(request.content, chat_response.output)
        faithfulness = ragas_scores["faithfulness"]
        answer_relevance = ragas_scores["answer_relevance"]
        context_precision = ragas_scores["context_precision"]
        similarity = ragas_scores.get("similarity")

        # Update Prometheus metrics
        faithfulness_gauge.set(faithfulness)
        answer_relevance_gauge.set(answer_relevance)
        context_precision_gauge.set(context_precision)
        if similarity is not None:
            similarity_gauge.set(similarity)

        # Threshold checks and logging
        if faithfulness < FAITHFULNESS_THRESHOLD or answer_relevance < RELEVANCE_THRESHOLD:
            logging.warning(
                f"Low RAGAS score: faithfulness={faithfulness}, answer_relevance={answer_relevance}, "
                f"question={request.content}, answer={chat_response.output}"
            )
            log_low_quality_response(request.content, chat_response.output, faithfulness, answer_relevance, None)

        return chat_response


@router.get(
    "/chats/{chatId}/responses/{responseId}/rating",
    response_model=schemas.RatingObject,
    response_model_exclude_none=True,
)
async def get_rating(
    chatId: str,  # pylint: disable=invalid-name
    responseId: str,  # pylint: disable=invalid-name
    user=fastapi.Depends(auth.require_employee_role),
) -> schemas.RatingObject:
    """Endpoint to get the rating of a specific response in a chat.

    Args:
        chatId (str): The ID of the chat.

    Raises:
        fastapi.HTTPException: If the chat is not found.
        fastapi.HTTPException: If the user is not allowed to rate the chat.
        fastapi.HTTPException: If the response is not found.

    Returns:
        schemas.RatingObject: The rating of the response.
    """
    user_obj = DbFeedbackInterface.get_user_by_email(email=user["email"])

    validation.validate_chat_existence_access(
        chat_id=chatId, meta_user_id=user_obj.meta_user_id
    )

    response_obj = DbFeedbackInterface.get_response_by_id(responseId)
    if response_obj is None:
        raise fastapi.HTTPException(status_code=404, detail="Response not found")

    if response_obj.rating is None:
        return schemas.RatingObject(rating=None)

    rating_obj = schemas.RatingObject.model_validate({"rating": response_obj.rating})
    return rating_obj


@router.post(
    "/chats/{chatId}/responses/{responseId}/rating",
    response_model=schemas.RatingObject,
    status_code=201,
)
async def create_rating(
    chatId: str,  # pylint: disable=invalid-name
    responseId: str,  # pylint: disable=invalid-name
    rating_request: schemas.RatingObject,
    user=fastapi.Depends(auth.require_employee_role),
) -> schemas.RatingObject:
    """Create a rating for a specific response in a chat.

    Args:
        chatId (str): The ID of the chat.
        responseId (str): The ID of the response.
        rating_request (schemas.RatingObject): The rating object containing the rating value.
        user (dict, optional): The authenticated user.

    Raises:
        fastapi.HTTPException: If the chat is not found.
        fastapi.HTTPException: If the user is not allowed to rate the chat.
        fastapi.HTTPException: If the response is not found.

    Returns:
        schemas.RatingObject: The created rating object.
    """
    user_obj = DbFeedbackInterface.get_user_by_email(email=user["email"])

    validation.validate_chat_existence_access(
        chat_id=chatId, meta_user_id=user_obj.meta_user_id
    )

    response_obj = DbFeedbackInterface.update_response_rating(
        responseId, rating_request.rating.value
    )
    if response_obj is None:
        raise fastapi.HTTPException(status_code=404, detail="Response not found")

    rating_obj = schemas.RatingObject.model_validate({"rating": response_obj.rating})
    return rating_obj


@router.put(
    "/chats/{chatId}/responses/{responseId}/rating",
    response_model=schemas.RatingObject,
    status_code=201,
)
async def update_rating(
    chatId: str,  # pylint: disable=invalid-name
    responseId: str,  # pylint: disable=invalid-name
    rating_request: schemas.RatingObject,
    user=fastapi.Depends(auth.require_employee_role),
):
    """Update the rating for a specific response in a chat.

    Args:
        chatId (str): The ID of the chat.
        responseId (str): The ID of the response.
        rating_request (schemas.RatingObject): The rating object containing the rating value.
        user (dict, optional): The authenticated user.

    Raises:
        fastapi.HTTPException: If the chat is not found.
        fastapi.HTTPException: If the user is not allowed to rate the chat.
        fastapi.HTTPException: If the response is not found.

    Returns:
        schemas.RatingObject: The updated rating object.
    """
    user_obj = DbFeedbackInterface.get_user_by_email(email=user["email"])

    validation.validate_chat_existence_access(
        chat_id=chatId, meta_user_id=user_obj.meta_user_id
    )

    response_obj = DbFeedbackInterface.update_response_rating(
        responseId, rating_request.rating.value
    )
    if response_obj is None:
        raise fastapi.HTTPException(status_code=404, detail="Response not found")

    rating_obj = schemas.RatingObject.model_validate({"rating": response_obj.rating})
    return rating_obj


@router.delete("/chats/{chatId}/responses/{responseId}/rating", status_code=204)
async def delete_rating(
    chatId: str,  # pylint: disable=invalid-name
    responseId: str,  # pylint: disable=invalid-name
    user=fastapi.Depends(auth.require_employee_role),
):
    """Delete the rating for a specific response in a chat.

    Args:
        chatId (str): The ID of the chat.
        responseId (str): The ID of the response.
        user (dict, optional): The authenticated user.

    Raises:
        fastapi.HTTPException: If the chat is not found.
        fastapi.HTTPException: If the user is not allowed to rate the chat.
        fastapi.HTTPException: If the response is not found.

    Returns:
        None
    """
    user_obj = DbFeedbackInterface.get_user_by_email(email=user["email"])

    validation.validate_chat_existence_access(
        chat_id=chatId, meta_user_id=user_obj.meta_user_id
    )

    response_obj = DbFeedbackInterface.update_response_rating(responseId, None)
    if response_obj is None:
        raise fastapi.HTTPException(status_code=404, detail="Response not found")

    return None


@router.post("/questions")
async def questions(
    request: schemas.QuestionsRequest, user=fastapi.Depends(auth.require_employee_role)
):
    path = "/search/question"
    search_question_url = config.settings.knowledge_graph_api_url + path
    async with httpx.AsyncClient() as client:
        post_data = {
            "question_history": request.question_history,
        }
        response = await client.post(search_question_url, json=post_data, timeout=20.0)
        response.raise_for_status()
        knowledge_graph_response = response.json()
        return knowledge_graph_response
