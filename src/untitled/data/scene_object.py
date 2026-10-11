from untitled.core.base import LOG
from untitled.data.natural_language_processor import NaturalLanguageProcessor

from enum import Enum
from spacy.tokens import Token
from dataclasses import dataclass
from typing import List

# NOTE: In data_agent.py, collected classes are: car, walker, traffic_light, stop_sign, static_car, static_trafficwarning, static

class SceneObjectType(Enum):
    """
    Enum for different types of scene objects.
    """
    UNKNOWN = "unknown"
    CAR = "car"
    MOTORBIKE = "motorbike"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    TRAFFIC_SIGN = "traffic_sign"
    TRAFFIC_LIGHT = "traffic_light"     # probably just stop sign
    ROAD = "road"
    DRIVING_LANE = "driving_lane"
    PARKING_LANE = "parking_lane"
    BIKE_LANE = "bike_lane"
    SHOULDER = "shoulder"
    SIDEWALK = "sidewalk"
    INTERSECTION = "intersection"
    JUNCTION = "junction"
    CONSTRUCTION = "construction"
    ACCIDENT = "accident"
    MISC = "misc"  # For any other scene objects that don't fit into the above categories

class SceneObjectModifierType(Enum):
    """
    Enum for different types of adjectives that can modify scene objects.
    """
    UNKNOWN = "unknown"
    COLOR = "color"
    POSITION = "position"

class ColorModifier(Enum):
    """
    Enum for different color modifiers.
    """
    NONE = "none"
    # Colors from webcolors.CSS2_HEX_TO_NAMES
    AQUA = "aqua"
    BLACK = "black"
    BLUE = "blue"
    FUCHSIA = "fuchsia"
    GREEN = "green"
    GRAY = "gray"
    LIME = "lime"
    MAROON = "maroon"
    NAVY = "navy"
    OLIVE = "olive"
    PURPLE = "purple"
    RED = "red"
    SILVER = "silver"
    TEAL = "teal"
    WHITE = "white"
    YELLOW = "yellow"

class PositionModifier(Enum):
    """
    Enum for different relative position modifiers.
    """
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BACK = "back"


def parse_color_modifier(token: Token) -> ColorModifier:
    """
    Parse the given token to determine what type of color modifier it is.
    Returns NONE type if the token is not a recognized color.
    """
    color_text = token.text.lower()
    if color_text.upper() in ColorModifier.__members__:
        return ColorModifier[color_text.upper()]
    return ColorModifier.NONE

# Different words can describe the same position relative to the ego vehicle.
_POSITION_MODIFIER_WORDS = {
    "left": PositionModifier.LEFT,
    "right": PositionModifier.RIGHT,
    "front": PositionModifier.FRONT,
    "ahead": PositionModifier.FRONT,
    "forward": PositionModifier.FRONT,
    "back": PositionModifier.BACK,
    "behind": PositionModifier.BACK,
    "rear": PositionModifier.BACK,
}
# These words identify the lane type; they are not additional modifiers.
_LANE_TYPE_WORDS = {"driving", "parking", "bike", "bicycle", "cycling", "biking"}


def parse_position_modifier(token: Token) -> PositionModifier:
    """
    Parse the given token to determine what type of position modifier it is.
    Returns NONE if it describes an action or a position relative to another object.
    """
    position_type = _POSITION_MODIFIER_WORDS.get(token.text.lower(), PositionModifier.NONE)
    if position_type == PositionModifier.NONE:
        return PositionModifier.NONE

    # 1. Exclude movement directions, such as "turn left" or "turn to the left".
    # Follow prepositions and "side" to find what the position phrase modifies.
    phrase_head = token.head
    while phrase_head.pos_ == "ADP" or phrase_head.text.lower() == "side":
        if phrase_head.head == phrase_head:
            break
        phrase_head = phrase_head.head

    movement_verbs = [
        "turn", "steer", "move", "drive", "go", "veer", "bear", "head", "merge",
    ]
    if phrase_head.pos_ == "VERB" and phrase_head.lemma_.lower() in movement_verbs:
        return PositionModifier.NONE

    # 2. Check the reference of the position phrase.
    # "In front of you" is ego-relative; "in front of the bus" is not.
    # For "left side", the reference may be attached to "side" instead of "left".
    position_phrase = token
    if token.dep_ == "amod" and token.head.text.lower() == "side":
        position_phrase = token.head

    for phrase_token in position_phrase.subtree:
        if phrase_token.dep_ == "poss":
            if phrase_token.text.lower() not in ["your", "our", "my"]:
                return PositionModifier.NONE
        if phrase_token.dep_ in ["pobj", "obj"] and phrase_token != token:
            if phrase_token.text.lower() not in ["you", "us", "me"]:
                return PositionModifier.NONE

    return position_type


@dataclass
class SceneObjectModifier:
    """
    Represents an adjective modifier for a scene object.
    """
    token: Token        # descriptive word within the modifier phrase
    type: SceneObjectModifierType
    value: None | ColorModifier | PositionModifier = None      # depends on the type

@dataclass
class SceneObject:
    """
    Represents a scene objects detected in a sentence.
    """
    token: Token        # noun
    type: SceneObjectType
    modifiers: List[SceneObjectModifier]

def create_scene_object(token: Token, object_type: SceneObjectType) -> SceneObject:
    """
    Creates a SceneObject instance from a noun, its type, and its modifiers.
    Includes adjective modifiers, spatial phrases, and descriptive predicates.
    """
    # Index modifiers by token so that a word reached through multiple dependency
    # paths is only counted once. Return them in sentence order afterwards.
    modifiers_by_token: dict[int, SceneObjectModifier] = {}

    def add_modifier(modifier_token: Token, allow_unknown: bool = False) -> None:
        """
        Classifies a modifier and adds it to the object.
        Unknown modifiers are kept only for adjective descriptions.
        """
        # "Parking" in "parking lane" identifies the object type.
        if token.text.lower() in ["lane", "lanes"]:
            if modifier_token.text.lower() in _LANE_TYPE_WORDS:
                return

        # Determine the modifier type, checking color before position.
        color_type = parse_color_modifier(modifier_token)
        position_type = parse_position_modifier(modifier_token)
        if color_type != ColorModifier.NONE:
            modifier = SceneObjectModifier(
                token=modifier_token, type=SceneObjectModifierType.COLOR, value=color_type,
            )
        elif position_type != PositionModifier.NONE:
            modifier = SceneObjectModifier(
                token=modifier_token, type=SceneObjectModifierType.POSITION, value=position_type,
            )
        elif allow_unknown and modifier_token.text.lower() not in _POSITION_MODIFIER_WORDS:
            modifier = SceneObjectModifier(token=modifier_token, type=SceneObjectModifierType.UNKNOWN)
        else:
            # A rejected position word should not become an unknown adjective.
            return

        modifiers_by_token[modifier_token.i] = modifier

    def collect_position_modifiers(phrase_token: Token) -> None:
        """
        Follows spatial phrases like "on the left" or "on your left side".
        Stops before unrelated nouns or verb clauses.
        """
        if phrase_token.text.lower() in _POSITION_MODIFIER_WORDS:
            add_modifier(phrase_token)

        for child in phrase_token.children:
            if child.dep_ not in ["prep", "pobj", "advmod", "amod"]:
                continue
            is_position_word = child.text.lower() in _POSITION_MODIFIER_WORDS
            is_phrase_connector = child.text.lower() in ["on", "in", "to", "at", "side"]
            if is_position_word or is_phrase_connector:
                collect_position_modifiers(child)

    # 1. Collect modifiers attached directly to the noun.
    for child in token.children:
        if child.dep_ == "amod":  # adjectival modifier
            add_modifier(child, allow_unknown=True)
        elif child.dep_ in ["prep", "advmod", "compound"]:
            collect_position_modifiers(child)

    # 2. Collect descriptions attached to the noun's predicate.
    # Examples: "the car is blue", "the car is on your right", and
    # "the car is parked on the left". Only the subject receives these modifiers.
    if token.dep_ in ["nsubj", "nsubjpass"]:
        for child in token.head.children:
            if child.dep_ in ["acomp", "attr"] and child.pos_ == "ADJ":
                add_modifier(child, allow_unknown=True)
            elif child.dep_ in ["prep", "advmod"]:
                collect_position_modifiers(child)

    # 3. Recover a spatial phrase immediately following the noun.
    # spaCy may attach "ahead" to "Follow" in "Follow the car ahead".
    # Requiring adjacency avoids assigning a distant phrase to the wrong object.
    next_token_index = token.i + 1
    if next_token_index < len(token.doc):
        next_token = token.doc[next_token_index]
        if next_token.head == token.head and next_token.dep_ in ["advmod", "prep"]:
            collect_position_modifiers(next_token)

    modifier_objects = [
        modifiers_by_token[token_index] for token_index in sorted(modifiers_by_token)
    ]
    return SceneObject(token=token, type=object_type, modifiers=modifier_objects)


def parse_scene_objects(sentence: str) -> List[SceneObject]:
    """
    Parses the input sentence and returns a list of scene objects (nouns) detected in the sentence.
    """
    nlp = NaturalLanguageProcessor.get()
    doc = nlp.process(sentence)
    scene_objects = []
    for token in doc:
        # Scene objects are nouns
        object_type: SceneObjectType = SceneObjectType.UNKNOWN
        if token.pos_ == "NOUN":
            # "Bike" in "bike lane" names the lane type, not another object.
            is_lane_description = token.dep_ in ["compound", "amod"]
            modifies_lane = token.head.text.lower() in ["lane", "lanes"]
            if is_lane_description and modifies_lane and token.text.lower() in _LANE_TYPE_WORDS:
                continue

            # Compound descriptions also identify "parking spot" and
            # "construction zone"; do not count the compound as another object.
            if token.dep_ == "compound":
                if token.head.lemma_.lower() == "spot" and token.text.lower() == "parking":
                    continue
                if token.head.lemma_.lower() == "zone" and token.text.lower() in ["construction", "work"]:
                    continue
            # Adjectives and compound nouns help identify the object type:
            # "construction zone", "parking spot", or "bike lane".
            # Spatial phrases like "on the left" are collected separately by
            # create_scene_object(), following prep/pobj dependencies as well.
            description_words = [
                child.text.lower() for child in token.children
                if child.dep_ in ["amod", "compound"]
            ]

            # Use the lemma to recognize plural nouns. Keep "parking" as written,
            # since its lemma can be "park", which has a different meaning.
            noun_text = token.lemma_.lower() or token.text.lower()
            if token.text.lower() == "parking":
                noun_text = "parking"
            
            # Detect type of object
            if noun_text in [
                "car", "cars", "vehicle", "vehicles", "automobile", "automobiles", "sedan", "sedans", "suv", "suvs", "truck", "trucks", "van", "vans", "bus", "buses", "mini", "minis", "taxi", "taxis"
            ]:
                object_type = SceneObjectType.CAR
            elif noun_text in ["motorcycle", "motorcycles", "motorbike", "motorbikes", "scooter", "scooters"]:
                object_type = SceneObjectType.MOTORBIKE
            elif noun_text in ["bicycle", "bicycles", "bike", "bikes", "cyclist", "cyclists"]:
                object_type = SceneObjectType.BICYCLE
            elif noun_text in ["pedestrian", "person"]:
                object_type = SceneObjectType.PEDESTRIAN
            elif noun_text in ["sign"]:
                object_type = SceneObjectType.TRAFFIC_SIGN
            elif noun_text in ["light"]:
                object_type = SceneObjectType.TRAFFIC_LIGHT
            elif noun_text in ["road", "street", "highway"]:
                object_type = SceneObjectType.ROAD
            elif noun_text in ["lane"]:
                if any(word in ["bike", "bicycle", "cycling", "biking"] for word in description_words):
                    object_type = SceneObjectType.BIKE_LANE
                elif "parking" in description_words:
                    object_type = SceneObjectType.PARKING_LANE
                else:
                    object_type = SceneObjectType.DRIVING_LANE
            elif noun_text in ["intersection"]:
                object_type = SceneObjectType.INTERSECTION
            elif noun_text in ["junction", "exit"]:
                object_type = SceneObjectType.JUNCTION
            elif noun_text in ["construction", "worker", "workers"]:
                object_type = SceneObjectType.CONSTRUCTION
            elif noun_text in ["accident"]:
                object_type = SceneObjectType.ACCIDENT
            elif noun_text in ["parking"]:
                object_type = SceneObjectType.PARKING_LANE
            elif noun_text in ["shoulder"]:
                object_type = SceneObjectType.SHOULDER
            elif noun_text in ["sidewalk", "pavement"]:
                object_type = SceneObjectType.SIDEWALK

            # Nouns that require descriptive words to determine their type
            elif noun_text in ["zone"]:
                if any(word in ["construction", "work"] for word in description_words):
                    object_type = SceneObjectType.CONSTRUCTION
            elif noun_text in ["spot"]:
                if "parking" in description_words:
                    object_type = SceneObjectType.PARKING_LANE

            # Unsupported scene objects
            elif noun_text in ["debris", "guardrail"]:
                object_type = SceneObjectType.MISC

            # Below are types to ignore, as they are not scene objects
            elif noun_text in ["m", "km", "m/s", "km/h", "mph", "speed", "velocity"]:
                continue
            elif noun_text in ["side", "left", "right", "front", "back", "middle"]:
                continue
            elif noun_text in ["bit", "moment", "gap", "room", "hurry"]:
                continue
            elif noun_text in ["nobody", "one"]:
                continue
            elif noun_text in ["U-turn", "U", "-", "turn", "moves"]:
                continue
            elif noun_text in ["house", "building"]: # They are scene objects but not related to driving
                continue
            elif noun_text in ["group", "traffic"]:
                continue

            # If type could not be resolved, skip it
            if object_type == SceneObjectType.UNKNOWN:
                LOG.warning(f"Unknown noun detected '{token.text}' in sentence '{sentence}'. It may be an unsupported scene object.")
                continue

            # Append object to list
            scene_objects.append(create_scene_object(token, object_type))

    return scene_objects
