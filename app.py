# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import traceback
from datetime import datetime, timezone
from aiohttp import web
from aiohttp.web import Request, Response, json_response
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
                                
                   
 
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.schema import Activity, ActivityTypes
from botbuilder.core import UserState, MemoryStorage
from bot import MyBot
from config import DefaultConfig
                                       

# Application configuration
CONFIG = DefaultConfig()

# Define your verification token for WhatsApp webhook validation
VERIFY_TOKEN = "8wb62gNUKNssAWigp5xRtbJhnmXZCqJnEUUEkuREdzDRvW55YuGUJQQJ99BCAC3pKaRAArohAAABAZBSk2cZ.F6PtzrIRxz9rvtufR0Y4MoQloM2fub1DNmbWyWj8H2R8pR4jjupeJQQJ99BCAC3pKaRAArohAAABAZBSYsNi"  # Replace with your secure token

# Create adapter.
                                                                         
SETTINGS = BotFrameworkAdapterSettings(CONFIG.APP_ID, CONFIG.APP_PASSWORD)
ADAPTER = BotFrameworkAdapter(SETTINGS)


# Catch-all for errors.
async def on_error(context: TurnContext, error: Exception):
                                                                    
                                                                                
                                 
    print(f"\n [on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()

    # Send a message to the user
    await context.send_activity("The bot encountered an error or bug.")
                                
    await context.send_activity("To continue to run this bot, please fix the bot source code.")
    
    # Send a trace activity if we're talking to the Bot Framework Emulator
    if context.activity.channel_id == "emulator":
                                                                
        trace_activity = Activity(
            label="TurnError",
            name="on_turn_error Trace",
            timestamp=datetime.now(timezone.utc),
                                         
            type=ActivityTypes.trace,
            value=f"{error}",
            value_type="https://www.botframework.com/schemas/error",
        )
                                                                                  
        await context.send_activity(trace_activity)


ADAPTER.on_turn_error = on_error

# Set up memory storage and user state
MEMORY = MemoryStorage()

                                       
USER_STATE = UserState(MEMORY)

                                                   
BOT = MyBot(USER_STATE)


# Main handler for both GET and POST requests on /api/messages
async def messages(req: Request) -> Response:
    # WhatsApp webhook verification (GET request)
    if req.method == "GET":
        mode = req.query.get("hub.mode")
        token = req.query.get("hub.verify_token")
        challenge = req.query.get("hub.challenge")
        
        if mode and token:
            if mode == "subscribe" and token == VERIFY_TOKEN:
                # Valid verification request: return the challenge token.
                return Response(text=challenge, status=200)
            else:
                # Token mismatch or invalid mode.
                return Response(status=403)
        else:
            # Missing required query parameters.
            return Response(status=400)
    
    # Message handling for POST requests (bot logic)
    if req.method == "POST":
        if "application/json" in req.headers.get("Content-Type", ""):
            body = await req.json()
        else:
            return Response(status=415)

        activity = Activity().deserialize(body)
        auth_header = req.headers.get("Authorization", "")
                                                                        
                               

        response = await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)
        if response:
            return json_response(data=response.body, status=response.status)
        return Response(status=201)

# Initialize the aiohttp application and add routes for GET and POST
def init_func(argv):
    APP = web.Application(middlewares=[aiohttp_error_middleware])
    APP.router.add_get("/api/messages", messages)
    APP.router.add_post("/api/messages", messages)
    return APP

                                                              
                                               

if __name__ == "__main__":
    APP = init_func(None)
    try:
        web.run_app(APP, host="0.0.0.0", port=CONFIG.PORT)
    except Exception as error:
        raise error
