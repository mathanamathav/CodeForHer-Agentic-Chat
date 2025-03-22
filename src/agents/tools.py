import json
import math
import os
import re
from datetime import datetime

import numexpr
import requests
from dotenv import load_dotenv
from langchain_core.tools import BaseTool, tool


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


def get_nearby_safe_places(location: str) -> list[dict]:
    """
    Get nearby places using ola API endpoint

    Useful for when you need to answer questions about nearby safe places.
    This tool is only for nearby safe places and nothing else. Only input
    location.

    Args:
        location (str): The location to search for nearby places

    Returns:
        str: A list of nearby places
    """
    load_dotenv()
    url = f"{os.getenv('BACKEND_URL')}/api/maps/get-latitude-longitude"
    payload = json.dumps({"address": location})
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer asdasd",
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response_json = response.json()
    latitude = response_json.get("latitude", "")
    longitude = response_json.get("longitude", "")

    url = f"{os.getenv('BACKEND_URL')}/api/maps/nearby-safe-spots"
    payload = json.dumps(
        {
            "current_location": {
                "latitude": latitude,
                "longitude": longitude,
                "address": location,
            },
            "radius": 5000,
            "rank_by": "distance",
        }
    )
    response = requests.request("POST", url, headers=headers, data=payload)
    response_json = response.json()
    nearby_places = response_json["predictions"]

    return nearby_places


nearby_places: BaseTool = tool(get_nearby_safe_places)
nearby_places.name = "Nearby_Places"


def get_current_location() -> dict:
    """
    Get current location based on IP address.

    Useful for when you need to determine the user's current location.
    This tool retrieves geolocation data based on the IP address.

    Returns:
        dict: A dictionary containing location information including address,
              latitude, longitude, city, country, etc.
    """
    load_dotenv()
    try:
        # Make request to IP geolocation API
        response = requests.get("https://ipinfo.io/json")
        if response.status_code != 200:
            raise ValueError(f"Failed to get location data: {response.status_code}")

        ip_data = response.json()

        # Extract location coordinates from the response (format: "latitude,longitude")
        if "loc" in ip_data:
            lat, lon = ip_data["loc"].split(",")

            # Format response
            location_data = {
                "ip": ip_data.get("ip", ""),
                "city": ip_data.get("city", ""),
                "region": ip_data.get("region", ""),
                "country": ip_data.get("country", ""),
                "postal": ip_data.get("postal", ""),
                "latitude": lat,
                "longitude": lon,
                "address": f"{ip_data.get('city', '')}, {ip_data.get('region', '')}, {ip_data.get('country', '')}",
            }
            return location_data
        else:
            raise ValueError("Location coordinates not found in the response")

    except Exception as e:
        raise ValueError(f"Error getting current location: {str(e)}")


current_location: BaseTool = tool(get_current_location)
current_location.name = "Current_Location"


def get_route(start: str, destination: str) -> list[dict]:
    """
    Get a route between two locations.

    Useful for when you need to find directions between two places.
    This tool returns a list of steps with navigation instructions.
    just get area name and city name from the start and destination

    Args:
        start (str): Starting location address or place name without country , state , pincode
        destination (str): Destination location address or place name without country , state , pincode

    Returns:
        list[dict]: A list of dictionaries containing route steps with:
                   - instruction: Text description of the step
                   - distance: Distance for this step
                   - duration: Estimated duration for this step
    """
    load_dotenv()
    try:
        url = f"{os.getenv('BACKEND_URL')}/api/maps/get-latitude-longitude"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer asdasd",
        }
        payload = json.dumps({"address": start})

        response = requests.request("POST", url, headers=headers, data=payload)
        response_json = response.json()
        start_latitude = response_json.get("latitude", "")
        start_longitude = response_json.get("longitude", "")

        payload = json.dumps({"address": destination})

        response = requests.request("POST", url, headers=headers, data=payload)
        response_json = response.json()
        destination_latitude = response_json.get("latitude", "")
        destination_longitude = response_json.get("longitude", "")

        url = f"{os.getenv('BACKEND_URL')}/api/maps/get-route"
        payload = json.dumps(
            {
                "origin": {
                    "latitude": start_latitude,
                    "longitude": start_longitude,
                    "address": start,
                },
                "destination": {
                    "latitude": destination_latitude,
                    "longitude": destination_longitude,
                    "address": destination,
                },
            }
        )
        response = requests.request("POST", url, headers=headers, data=payload)
        response_json = response.json()

        routes = response_json["routes"]

        route_steps = []
        for leg in routes[0].get("legs", []):
            for step in leg.get("steps", []):
                # Create RouteStep object
                route_step = {
                    "instructions": step.get("instructions", ""),
                    "distance": step.get("readable_distance", ""),
                    "duration": step.get("readable_duration", ""),
                }
                route_steps.append(route_step)

        url = f"{os.getenv('BACKEND_URL')}/api/llm/route-safety"
        payload = json.dumps({"route_steps": route_steps})
        response = requests.request("POST", url, headers=headers, data=payload)
        response_json = response.json()

        response_data = {
            "route_steps": route_steps,
            "safety_tips": response_json.get("safety_tips", ""),
        }

        return response_data

    except Exception as e:
        raise ValueError(f"Error getting route: {str(e)}")


route: BaseTool = tool(get_route)
route.name = "get_route"


def send_sos_alert(location: str, custom_message: str = "Help! I am in danger.") -> str:
    """
    Send SOS alert to emergency services.

    Useful for when you need to send an SOS alert to emergency services.
    This tool sends an SOS alert to emergency services.

    Args:
        location (str): Location address or place name without country , state , pincode
        custom_message (str): Custom message to send to emergency services
    Returns:
        str: A message indicating the SOS alert was sent successfully.
    """
    load_dotenv()
    try:
        url = f"{os.getenv('BACKEND_URL')}/api/maps/get-latitude-longitude"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer asdasd",
        }
        payload = json.dumps({"address": location})

        response = requests.request("POST", url, headers=headers, data=payload)
        response_json = response.json()
        latitude = response_json.get("latitude", "")
        longitude = response_json.get("longitude", "")

        url = f"{os.getenv('BACKEND_URL')}/api/sos/send-alert"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer asdasd",
        }

        payload = json.dumps(
            {
                "user_id": "67de72427e1f7f041d589ba7",
                "timestamp": datetime.now().isoformat(),
                "location": {
                    "latitude": latitude,
                    "longitude": longitude,
                    "address": location,
                },
                "message": custom_message,
            }
        )

        response = requests.request("POST", url, headers=headers, data=payload)
        response_json = response.json()
        return response_json.get("message", "")

    except Exception as e:
        raise ValueError(f"Error sending SOS alert: {str(e)}")


sos_alert: BaseTool = tool(send_sos_alert)
sos_alert.name = "sos_alert"

if __name__ == "__main__":
    print(
        send_sos_alert(
            location="Axis Bank Limited, Alapakkam, Chengalpattu, Tamil Nadu"
        )
    )
