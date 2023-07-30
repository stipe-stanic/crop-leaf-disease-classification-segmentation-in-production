import json
import traceback

from fastapi import Request, status
from fastapi.logger import logger
from fastapi.responses import JSONResponse


def get_error_response(request: Request, ex: Exception) -> dict:
    """Generate a generic error response.

    :param request: The request object.
    :param ex: The exception object.
    :return: The error response dictionary.
    """

    error_response = {
        "error": True,
        "message": str(ex)
    }

    error_response["traceback"] = "".join(
        traceback.format_exception(
            type(ex), value=ex, tb=ex.__traceback__
        )
    )

    return error_response


async def python_exception_handler(request: Request, ex: Exception):
    """Handle any internal error.

   :param request: The request object.
   :param ex: The exception object.
   :return: The JSON response with the error details.
   """

    # Logs request information
    logger.error('Request info:\n' + json.dumps({
        "host": request.client.host,  # ip address or hostname
        "method": request.method,
        "url": str(request.url),
        "headers": str(request.headers),
        "path_params": str(request.path_params),
        "query_params": str(request.query_params),
        "cookies": str(request.cookies)
    }, indent=4))

    # Generates JSON response with error details
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=get_error_response(request, ex)
    )