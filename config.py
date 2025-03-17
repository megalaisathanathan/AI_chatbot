#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

class DefaultConfig:
    """ Bot Configuration """

    PORT = 3978
    APP_ID = os.environ.get("MicrosoftAppId", "8f09db99-9562-48d3-ae5a-6e7865d59b73")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "W1R8Q~jCdbUAvxxUaD2f8yzLgWaR-MlmUfibBb_8")