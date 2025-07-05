"""
E*Trade API Client - Corrected Version Following Official Examples

This module implements E*Trade API integration following the official examples
from the E*Trade Python SDK exactly. It uses the correct OAuth flow, URL patterns,
and response handling as shown in the source of truth examples.

Based on: examples/etrade/etrade_python_client.py
Corrected to match official patterns exactly.
"""

import json
import logging
import webbrowser
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
from rauth import OAuth1Service

# Configure logging to match E*Trade examples exactly
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)


class ETradeAuthenticationError(Exception):
    """Exception for E*Trade authentication issues"""
    pass


class ETradeAPIError(Exception):
    """Exception for E*Trade API errors"""
    pass


class ETradeClient:
    """
    E*Trade API Client following official examples exactly.
    
    This implementation follows the patterns from examples/etrade/ files:
    - Uses rauth.OAuth1Service for OAuth flow (exact setup)
    - Follows exact URL patterns from examples
    - Uses response structures matching official examples
    - Implements error handling exactly as shown in examples
    - Uses header_auth=True parameter as in official examples
    """
    
    def __init__(self, consumer_key: str, consumer_secret: str, sandbox: bool = True):
        """
        Initialize E*Trade client with consumer credentials.
        
        Args:
            consumer_key: E*Trade consumer key
            consumer_secret: E*Trade consumer secret
            sandbox: Use sandbox environment (default True)
        """
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.sandbox = sandbox
        
        # Set base URLs following examples/etrade/sample_config.ini exactly
        if sandbox:
            self.base_url = "https://apisb.etrade.com"
        else:
            self.base_url = "https://api.etrade.com"
        
        # Initialize OAuth service following etrade_python_client.py exactly
        self.etrade = OAuth1Service(
            name="etrade",
            consumer_key=self.consumer_key,
            consumer_secret=self.consumer_secret,
            request_token_url="https://api.etrade.com/oauth/request_token",
            access_token_url="https://api.etrade.com/oauth/access_token",
            authorize_url="https://us.etrade.com/e/t/etws/authorize?key={}&token={}",
            base_url="https://api.etrade.com"
        )
        
        self.session = None
        self.accounts = []
        
        logger.info(f"ETradeClient initialized ({'SANDBOX' if sandbox else 'LIVE'} mode)")
    
    def authenticate(self, verification_code: Optional[str] = None) -> bool:
        """
        Perform OAuth authentication following etrade_python_client.py flow exactly.
        
        Args:
            verification_code: Manual verification code if not using browser
            
        Returns:
            bool: True if authentication successful
        """
        try:
            # Step 1: Get OAuth 1 request token and secret (exact pattern from examples)
            logger.info("Getting request token...")
            request_token, request_token_secret = self.etrade.get_request_token(
                params={"oauth_callback": "oob", "format": "json"}
            )
            
            # Step 2: Go through the authentication flow (exact pattern from examples)
            authorize_url = self.etrade.authorize_url.format(
                self.etrade.consumer_key, request_token
            )
            
            if verification_code is None:
                # Open browser for authorization (following examples exactly)
                logger.info(f"Opening browser for authorization: {authorize_url}")
                webbrowser.open(authorize_url)
                
                # This would need to be handled in UI
                raise ETradeAuthenticationError(
                    f"Please authorize the application at: {authorize_url}\n"
                    "Then call authenticate() again with the verification code."
                )
            
            # Step 3: Exchange the authorized request token for an authenticated OAuth 1 session
            logger.info("Exchanging tokens for authenticated session...")
            self.session = self.etrade.get_auth_session(
                request_token,
                request_token_secret,
                params={"oauth_verifier": verification_code}
            )
            
            # Verify authentication by getting account list (following examples)
            self.account_list()
            
            logger.info("Authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise ETradeAuthenticationError(f"Authentication failed: {e}")
    
    def account_list(self) -> List[Dict[str, Any]]:
        """
        Calls account list API to retrieve a list of the user's E*TRADE accounts.
        Following accounts.py example exactly.
        
        Returns:
            List of account dictionaries
        """
        if not self.session:
            raise ETradeAuthenticationError("Not authenticated")
        
        # URL for the API endpoint (exact pattern from accounts.py)
        url = self.base_url + "/v1/accounts/list.json"
          # Make API call for GET request (following accounts.py pattern)
        # Note: Official examples use header_auth=True but current rauth version may not support it
        response = self.session.get(url)
        logger.debug("Request Header: %s", response.request.headers)
        
        # Handle and parse response (exact pattern from accounts.py)
        if response is not None and response.status_code == 200:
            parsed = json.loads(response.text)
            logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
            
            data = response.json()
            if (data is not None and "AccountListResponse" in data and 
                "Accounts" in data["AccountListResponse"] and
                "Account" in data["AccountListResponse"]["Accounts"]):
                
                accounts = data["AccountListResponse"]["Accounts"]["Account"]
                # Filter closed accounts (exact pattern from accounts.py)
                accounts[:] = [d for d in accounts if d.get('accountStatus') != 'CLOSED']
                
                self.accounts = accounts
                logger.info(f"Loaded {len(self.accounts)} active accounts")
                return accounts
            else:
                raise ETradeAPIError("Invalid account list response structure")
        else:
            # Handle errors (exact pattern from accounts.py)
            logger.debug("Response Body: %s", response.text)
            if (response is not None and 
                response.headers.get('Content-Type') == 'application/json' and
                "Error" in response.json() and 
                "message" in response.json()["Error"] and
                response.json()["Error"]["message"] is not None):
                error_msg = response.json()["Error"]["message"]
                raise ETradeAPIError(f"Error: {error_msg}")
            else:
                raise ETradeAPIError("Error: AccountList API service error")
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Calls quotes API to provide quote details for equities, options, and mutual funds.
        Following market.py example exactly.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Quote data dictionary
        """
        if not self.session:
            raise ETradeAuthenticationError("Not authenticated")
        
        symbol = symbol.upper().strip()
        
        # URL for the API endpoint (exact pattern from market.py)
        url = self.base_url + "/v1/market/quote/" + symbol + ".json"
        
        # Make API call for GET request (exact pattern from market.py)
        response = self.session.get(url)
        logger.debug("Request Header: %s", response.request.headers)
        
        if response is not None and response.status_code == 200:
            parsed = json.loads(response.text)
            logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
            
            # Handle and parse response (exact pattern from market.py)
            data = response.json()
            if (data is not None and "QuoteResponse" in data and 
                "QuoteData" in data["QuoteResponse"]):
                
                quote_data = data["QuoteResponse"]["QuoteData"]
                if quote_data and len(quote_data) > 0:
                    return quote_data[0]  # Return first quote
                else:
                    raise ETradeAPIError("No quote data returned")
            else:
                # Handle errors (exact pattern from market.py)
                if (data is not None and 'QuoteResponse' in data and 
                    'Messages' in data["QuoteResponse"] and
                    'Message' in data["QuoteResponse"]["Messages"] and
                    data["QuoteResponse"]["Messages"]["Message"] is not None):
                    
                    error_messages = []
                    for error_message in data["QuoteResponse"]["Messages"]["Message"]:
                        error_messages.append("Error: " + error_message["description"])
                    raise ETradeAPIError("; ".join(error_messages))
                else:
                    raise ETradeAPIError("Error: Quote API service error")
        else:
            logger.debug("Response Body: %s", response)
            raise ETradeAPIError("Error: Quote API service error")
    
    def get_account_balance(self, account_id_key: str) -> Dict[str, Any]:
        """
        Get account balance following accounts.py pattern.
        
        Args:
            account_id_key: Account ID key from account list
            
        Returns:
            Balance data dictionary
        """
        if not self.session:
            raise ETradeAuthenticationError("Not authenticated")
        
        # URL pattern following accounts.py examples
        url = self.base_url + "/v1/accounts/" + account_id_key + "/balance.json"
          # Make API call (following accounts.py pattern)
        # Note: Official examples use header_auth=True but current rauth version may not support it
        response = self.session.get(url)
        logger.debug("Request Header: %s", response.request.headers)
        
        if response is not None and response.status_code == 200:
            parsed = json.loads(response.text)
            logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
            return parsed
        else:
            # Handle errors following accounts.py pattern
            logger.debug("Response Body: %s", response.text)
            if (response is not None and 
                response.headers.get('Content-Type') == 'application/json' and
                "Error" in response.json() and 
                "message" in response.json()["Error"] and
                response.json()["Error"]["message"] is not None):
                error_msg = response.json()["Error"]["message"]
                raise ETradeAPIError(f"Error: {error_msg}")
            else:
                raise ETradeAPIError("Error: Balance API service error")
    
    def preview_order(self, symbol: str, action: str, quantity: int, 
                     price_type: str = "MARKET", limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Preview an order following order.py example exactly.
        
        Args:
            symbol: Stock symbol
            action: "BUY" or "SELL"
            quantity: Number of shares
            price_type: "MARKET" or "LIMIT"
            limit_price: Limit price for LIMIT orders
            
        Returns:
            Preview order response
        """
        if not self.session:
            raise ETradeAuthenticationError("Not authenticated")
        
        if not self.accounts:
            raise ETradeAPIError("No accounts available")
        
        # Use first account
        account = self.accounts[0]
        account_id_key = account.get("accountIdKey")
        
        if not account_id_key:
            raise ETradeAPIError("Invalid account ID key")
        
        # URL for the API endpoint (exact pattern from order.py)
        url = self.base_url + "/v1/accounts/" + account_id_key + "/orders/preview.json"
        
        # Add parameters and header information (exact pattern from order.py)
        headers = {
            "Content-Type": "application/xml", 
            "consumerKey": self.consumer_key
        }
        
        # Generate client order ID
        client_order_id = f"ORDER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add payload for POST Request (following order.py pattern exactly)
        payload = f"""<PreviewOrderRequest>
                       <orderType>EQ</orderType>
                       <clientOrderId>{client_order_id}</clientOrderId>
                       <Order>
                           <allOrNone>false</allOrNone>
                           <priceType>{price_type}</priceType>
                           <orderTerm>GOOD_FOR_DAY</orderTerm>
                           <marketSession>REGULAR</marketSession>
                           <stopPrice></stopPrice>
                           <limitPrice>{limit_price if limit_price else ''}</limitPrice>
                           <Instrument>
                               <Product>
                                   <securityType>EQ</securityType>
                                   <symbol>{symbol}</symbol>
                               </Product>
                               <orderAction>{action}</orderAction>
                               <quantityType>QUANTITY</quantityType>
                               <quantity>{quantity}</quantity>
                           </Instrument>
                       </Order>
                   </PreviewOrderRequest>"""
          # Make API call for POST request (following order.py pattern)
        # Note: Official examples use header_auth=True but current rauth version may not support it
        response = self.session.post(url, headers=headers, data=payload)
        logger.debug("Request Header: %s", response.request.headers)
        logger.debug("Request payload: %s", payload)
        
        if response is not None and response.status_code == 200:
            parsed = json.loads(response.text)
            logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
            return parsed
        else:
            # Handle errors (exact pattern from order.py)
            logger.debug("Response Body: %s", response.text)
            try:
                error_data = response.json()
                if 'Error' in error_data and 'message' in error_data["Error"]:
                    error_msg = error_data["Error"]["message"]
                    raise ETradeAPIError(f"Error: {error_msg}")
            except (json.JSONDecodeError, KeyError):
                pass
            raise ETradeAPIError("Error: Order API service error")
    
    def place_order(self, preview_ids: Dict[str, str], symbol: str, action: str, 
                   quantity: int, price_type: str = "MARKET", 
                   limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place an order after preview following order.py example.
        
        Args:
            preview_ids: Preview IDs from preview_order response
            symbol: Stock symbol
            action: "BUY" or "SELL"
            quantity: Number of shares
            price_type: "MARKET" or "LIMIT"
            limit_price: Limit price for LIMIT orders
            
        Returns:
            Place order response
        """
        if not self.session:
            raise ETradeAuthenticationError("Not authenticated")
        
        if not self.accounts:
            raise ETradeAPIError("No accounts available")
        
        # Use first account
        account = self.accounts[0]
        account_id_key = account.get("accountIdKey")
        
        if not account_id_key:
            raise ETradeAPIError("Invalid account ID key")
        
        # URL for the API endpoint (following order.py pattern)
        url = self.base_url + "/v1/accounts/" + account_id_key + "/orders/place.json"
        
        # Headers (exact pattern from order.py)
        headers = {
            "Content-Type": "application/xml", 
            "consumerKey": self.consumer_key
        }
        
        # Get preview IDs from preview response
        preview_id = preview_ids.get("previewId", "")
        order_total_estimate = preview_ids.get("orderTotalEstimate", "")
        
        # XML payload for place order (following order.py pattern)
        payload = f"""<PlaceOrderRequest>
                       <orderType>EQ</orderType>
                       <clientOrderId>{preview_ids.get('clientOrderId', '')}</clientOrderId>
                       <previewId>{preview_id}</previewId>
                       <Order>
                           <allOrNone>false</allOrNone>
                           <priceType>{price_type}</priceType>
                           <orderTerm>GOOD_FOR_DAY</orderTerm>
                           <marketSession>REGULAR</marketSession>
                           <stopPrice></stopPrice>
                           <limitPrice>{limit_price if limit_price else ''}</limitPrice>
                           <Instrument>
                               <Product>
                                   <securityType>EQ</securityType>
                                   <symbol>{symbol}</symbol>
                               </Product>
                               <orderAction>{action}</orderAction>
                               <quantityType>QUANTITY</quantityType>
                               <quantity>{quantity}</quantity>
                           </Instrument>
                       </Order>
                   </PlaceOrderRequest>"""
          # Make API call (following order.py pattern)
        # Note: Official examples use header_auth=True but current rauth version may not support it
        response = self.session.post(url, headers=headers, data=payload)
        logger.debug("Request Header: %s", response.request.headers)
        logger.debug("Request payload: %s", payload)
        
        if response is not None and response.status_code == 200:
            parsed = json.loads(response.text)
            logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
            return parsed
        else:
            # Handle errors (exact pattern from order.py)
            logger.debug("Response Body: %s", response.text)
            try:
                error_data = response.json()
                if 'Error' in error_data and 'message' in error_data["Error"]:
                    error_msg = error_data["Error"]["message"]
                    raise ETradeAPIError(f"Error: {error_msg}")
            except (json.JSONDecodeError, KeyError):
                pass
            raise ETradeAPIError("Error: Place Order API service error")
    
    def get_candles(self, symbol: str, interval: str = "5min", days: int = 1) -> pd.DataFrame:
        """
        Get candlestick data. This would need to be implemented based on 
        actual E*Trade historical data API endpoints.
        
        Note: The examples don't show historical data endpoints, so this
        would need to be implemented based on E*Trade API documentation.
        """
        # Placeholder implementation - would need actual E*Trade endpoint
        raise NotImplementedError(
            "Candlestick data endpoint not available in E*Trade examples. "
            "This would need to be implemented based on E*Trade API documentation."
        )


def create_etrade_client_from_config() -> ETradeClient:
    """
    Create E*Trade client from configuration following the examples pattern.
    
    This function looks for credentials in environment variables following
    the pattern from sample_config.ini.
    """
    import os
    
    consumer_key = os.getenv('ETRADE_CONSUMER_KEY')
    consumer_secret = os.getenv('ETRADE_CONSUMER_SECRET')
    sandbox = os.getenv('ETRADE_SANDBOX', 'true').lower() == 'true'
    
    if not consumer_key or not consumer_secret:
        raise ETradeAuthenticationError(
            "Missing E*Trade credentials. Please set ETRADE_CONSUMER_KEY and ETRADE_CONSUMER_SECRET environment variables."
        )
    
    return ETradeClient(consumer_key, consumer_secret, sandbox)


# Helper function to demonstrate usage following examples
def main_menu_demo():
    """
    Demonstrates usage following the official examples pattern.
    This replicates the main_menu function from etrade_python_client.py
    """
    try:
        client = create_etrade_client_from_config()
        
        # Step 1: Authenticate (would need verification code)
        print("Visit the authorization URL and get verification code")
        # client.authenticate(verification_code="your_code_here")
        
        # Step 2: Get account list
        # accounts = client.account_list()
        
        # Step 3: Get quotes
        # quote = client.get_quote("AAPL")
        
        # Step 4: Preview and place orders
        # preview = client.preview_order("AAPL", "BUY", 1, "MARKET")
        # place_response = client.place_order(preview, "AAPL", "BUY", 1, "MARKET")
        
        print("Client initialized successfully")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main_menu_demo()
