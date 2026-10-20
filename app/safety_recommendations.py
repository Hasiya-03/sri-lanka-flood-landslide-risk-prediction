
def get_safety_recommendations(disaster_type, risk_level):
    recommendations = {
        "flood": {
            "LOW": [
                "Monitor local weather forecasts.",
                "Keep drainage systems clear."
            ],
            "MEDIUM": [
                "Prepare an emergency supply kit.",
                "Avoid travelling through flood-prone roads.",
                "Monitor official flood warnings."
            ],
            "HIGH": [
                "Avoid floodwater and submerged roads.",
                "Move to safer ground when advised by authorities.",
                "Follow official evacuation instructions."
            ]
        },
        "landslide": {
            "LOW": [
                "Monitor rainfall conditions.",
                "Observe changes in surrounding slopes."
            ],
            "MEDIUM": [
                "Watch for ground cracks and leaning trees.",
                "Stay alert during prolonged heavy rainfall.",
                "Review official landslide advisories."
            ],
            "HIGH": [
                "Stay away from unstable slopes.",
                "If signs of slope movement appear, move to safety.",
                "Follow official evacuation instructions."
            ]
        }
    }

    return recommendations.get(
        disaster_type.lower(), {}
    ).get(risk_level.upper(), [])
