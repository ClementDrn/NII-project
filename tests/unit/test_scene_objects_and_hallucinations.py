from typing import List

import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab

from untitled.data.hallucinations import check_hallucinations_in_instruction, parse_scene_objects
from untitled.data.scene_object import (
    ColorModifier,
    PositionModifier,
    SceneObject,
    SceneObjectModifier,
    SceneObjectModifierType,
    SceneObjectType,
    create_scene_object,
    parse_color_modifier,
    parse_position_modifier,
)


def test_parse_scene_objects():
    a: str = "Turn left at the intersection ahead."
    b: str = "Follow the car ahead and don't stop at the junction."
    c: str = "Accelerate to 50 km/h."
    d: str = "Increase speed to 50 km/h."
    e: str = "Be careful, a pedestrian is crossing!"

    a_objects: List[SceneObject] = parse_scene_objects(a)
    b_objects: List[SceneObject] = parse_scene_objects(b)
    c_objects: List[SceneObject] = parse_scene_objects(c)
    d_objects: List[SceneObject] = parse_scene_objects(d)
    e_objects: List[SceneObject] = parse_scene_objects(e)

    # Check number of objects detected in each sentence
    # And check that the objects are correctly identified
    assert len(a_objects) == 1
    assert a_objects[0].token.text == "intersection"
    assert a_objects[0].type == SceneObjectType.INTERSECTION

    assert len(b_objects) == 2
    assert b_objects[0].token.text == "car"
    assert b_objects[0].type == SceneObjectType.CAR
    assert b_objects[1].token.text == "junction"
    assert b_objects[1].type == SceneObjectType.JUNCTION

    assert len(c_objects) == 0

    assert len(d_objects) == 0

    assert len(e_objects) == 1
    assert e_objects[0].token.text == "pedestrian"
    assert e_objects[0].type == SceneObjectType.PEDESTRIAN


def test_parse_scene_objects_with_adjectives():
    sentence: str = "The red car is parked next to the blue bicycle."
    objects: List[SceneObject] = parse_scene_objects(sentence)

    assert len(objects) == 2

    # Check the first object (car)
    assert objects[0].token.text == "car"
    assert objects[0].type == SceneObjectType.CAR
    assert len(objects[0].modifiers) == 1
    assert objects[0].modifiers[0].type == SceneObjectModifierType.COLOR
    assert objects[0].modifiers[0].value == ColorModifier.RED

    # Check the second object (bicycle)
    assert objects[1].token.text == "bicycle"
    assert objects[1].type == SceneObjectType.BICYCLE
    assert len(objects[1].modifiers) == 1
    assert objects[1].modifiers[0].type == SceneObjectModifierType.COLOR
    assert objects[1].modifiers[0].value == ColorModifier.BLUE


def test_parse_scene_objects_with_no_scene_objects():
    sentence: str = "Follow the alien."
    objects: List[SceneObject] = parse_scene_objects(sentence)

    assert len(objects) == 0



def dependency_doc(words, heads, deps, pos, lemmas=None):
    return Doc(Vocab(), words=words, heads=heads, deps=deps, pos=pos,
               lemmas=lemmas or [word.lower() for word in words])


@pytest.mark.parametrize('word,expected', [
    ('left', PositionModifier.LEFT), ('right', PositionModifier.RIGHT),
    ('front', PositionModifier.FRONT), ('ahead', PositionModifier.FRONT),
    ('behind', PositionModifier.BACK), ('rear', PositionModifier.BACK),
    ('back', PositionModifier.BACK), ('forward', PositionModifier.FRONT),
])
def test_position_words(word, expected):
    doc = dependency_doc(['car', word], [0, 0], ['ROOT', 'advmod'], ['NOUN', 'ADV'])
    assert parse_position_modifier(doc[1]) == expected
    assert create_scene_object(doc[0], SceneObjectType.CAR).modifiers[0].value == expected


def test_dependency_spatial_and_action_contexts():
    spatial = dependency_doc(['car', 'on', 'your', 'left'], [0, 0, 3, 1],
                             ['ROOT', 'prep', 'poss', 'pobj'], ['NOUN', 'ADP', 'PRON', 'NOUN'])
    assert [m.value for m in create_scene_object(spatial[0], SceneObjectType.CAR).modifiers] == [PositionModifier.LEFT]
    action = dependency_doc(['car', 'turning', 'left'], [0, 0, 1],
                            ['ROOT', 'acl', 'advmod'], ['NOUN', 'VERB', 'ADV'], ['car', 'turn', 'left'])
    assert parse_position_modifier(action[2]) == PositionModifier.NONE
    assert not create_scene_object(action[0], SceneObjectType.CAR).modifiers
    relative = dependency_doc(['car', 'behind', 'bus'], [0, 0, 1],
                              ['ROOT', 'prep', 'pobj'], ['NOUN', 'ADP', 'NOUN'])
    assert parse_position_modifier(relative[1]) == PositionModifier.NONE
    assert not create_scene_object(relative[0], SceneObjectType.CAR).modifiers


def test_color_and_unknown_modifier():
    doc = dependency_doc(['Bright', 'RED', 'car'], [2, 2, 2],
                         ['amod', 'amod', 'ROOT'], ['ADJ', 'ADJ', 'NOUN'])
    assert parse_color_modifier(doc[1]) == ColorModifier.RED
    modifiers = create_scene_object(doc[2], SceneObjectType.CAR).modifiers
    assert [m.type for m in modifiers] == [SceneObjectModifierType.UNKNOWN, SceneObjectModifierType.COLOR]
    assert len({m.token.i for m in modifiers}) == 2


@pytest.mark.parametrize('sentence,kind,values', [
    ('Follow the red car ahead.', SceneObjectType.CAR, [ColorModifier.RED, PositionModifier.FRONT]),
    ('The car is on your right.', SceneObjectType.CAR, [PositionModifier.RIGHT]),
    ('Follow the car on the left.', SceneObjectType.CAR, [PositionModifier.LEFT]),
    ('Stop in the construction zone.', SceneObjectType.CONSTRUCTION, []),
    ('Use the parking spot.', SceneObjectType.PARKING_LANE, []),
    ('The car in front of you is blue.', SceneObjectType.CAR, [PositionModifier.FRONT, ColorModifier.BLUE]),
    ('The car is parked on the left.', SceneObjectType.CAR, [PositionModifier.LEFT]),
    ('Follow the car turning left.', SceneObjectType.CAR, []),
    ('Follow the car behind the bus.', SceneObjectType.CAR, []),
    ('Turn left at the intersection ahead.', SceneObjectType.INTERSECTION, [PositionModifier.FRONT]),
    ('Use the left bike lane.', SceneObjectType.BIKE_LANE, [PositionModifier.LEFT]),
    ('Use the parking lane.', SceneObjectType.PARKING_LANE, []),
    ('Use the driving lane.', SceneObjectType.DRIVING_LANE, []),
])
def test_modifier_integration(sentence, kind, values):
    objects = parse_scene_objects(sentence)
    assert objects[0].type == kind
    assert [m.value for m in objects[0].modifiers] == values
    if kind in {SceneObjectType.BIKE_LANE, SceneObjectType.PARKING_LANE, SceneObjectType.DRIVING_LANE}:
        assert len(objects) == 1


@pytest.fixture
def metric_objects(monkeypatch):
    def install(kind=SceneObjectType.CAR, colors=(), positions=(), unknown=False, repeats=1):
        doc = dependency_doc(['car'], [0], ['ROOT'], ['NOUN'])
        modifiers = [SceneObjectModifier(doc[0], SceneObjectModifierType.COLOR, color) for color in colors]
        modifiers += [SceneObjectModifier(doc[0], SceneObjectModifierType.POSITION, position) for position in positions]
        if unknown:
            modifiers.append(SceneObjectModifier(doc[0], SceneObjectModifierType.UNKNOWN))
        objects = [SceneObject(doc[0], kind, modifiers) for _ in range(repeats)]
        monkeypatch.setattr('untitled.data.hallucinations.parse_scene_objects', lambda _: objects)
    return install


@pytest.mark.parametrize('kind,actor', [
    (SceneObjectType.CAR, {'class': 'car'}),
    (SceneObjectType.CAR, {'class': 'static_car'}),
    (SceneObjectType.PEDESTRIAN, {'class': 'walker'}),
    (SceneObjectType.TRAFFIC_LIGHT, {'class': 'traffic_light'}),
    (SceneObjectType.TRAFFIC_SIGN, {'class': 'stop_sign'}),
    (SceneObjectType.BICYCLE, {'class': 'car', 'base_type': 'bicycle'}),
    (SceneObjectType.MOTORBIKE, {'class': 'car', 'base_type': 'motorcycle'}),
])
def test_actor_matching(metric_objects, kind, actor):
    metric_objects(kind)
    assert check_hallucinations_in_instruction('', {'scene': [actor]})['object_hallucination_count'] == 0
    assert check_hallucinations_in_instruction('', {'scene': []})['object_hallucination_count'] == 1


def test_metrics_joint_match_counts_and_missing_evidence(metric_objects):
    metric_objects(colors=[ColorModifier.RED], positions=[PositionModifier.FRONT], unknown=True, repeats=2)
    scene = [{'class': 'car', 'color_name': 'red', 'position': [-10, 0]},
             {'class': 'car', 'color_name': 'blue', 'position': [10, 0]}]
    assert check_hallucinations_in_instruction('', {'scene': scene}) == {
        'object_count': 2, 'modifier_count': 6, 'object_hallucination_count': 0,
        'modifier_hallucination_count': 2,
    }
    assert check_hallucinations_in_instruction('', {})['modifier_hallucination_count'] == 0
    assert check_hallucinations_in_instruction('', {'scene': [{'class': 'car'}]})['modifier_hallucination_count'] == 0
    absent = check_hallucinations_in_instruction('', {'scene': [{'class': 'ego_car'}]})
    assert absent['object_hallucination_count'] == 2
    assert absent['modifier_hallucination_count'] == 0
    assert check_hallucinations_in_instruction('', {'scene_objects': [], 'scene': scene})['object_hallucination_count'] == 2


@pytest.mark.parametrize('position,expected', [([1, 2], 0), ([-1, -2], 2), ([0, 0], 2), ([], 0), ([None, 'bad'], 0)])
def test_position_metrics(metric_objects, position, expected):
    metric_objects(positions=[PositionModifier.FRONT, PositionModifier.RIGHT])
    assert check_hallucinations_in_instruction('', {'scene': [{'class': 'car', 'position': position}]})['modifier_hallucination_count'] == expected


@pytest.mark.parametrize("ego_state, light_state, expected_count", [
    ("Green", "Red", 1),
    ("Red", "Green", 0),
    (None, "Green", 0),
])
def test_traffic_light_uses_ego_state(metric_objects, ego_state, light_state, expected_count):
    metric_objects(SceneObjectType.TRAFFIC_LIGHT, colors=[ColorModifier.RED])
    metadata = {"scene": [
        {"class": "ego_info", "traffic_light_state": ego_state},
        {"class": "traffic_light", "state": light_state},
    ]}

    # Nearby lights can control other lanes. Use only the ego vehicle's signal.
    result = check_hallucinations_in_instruction("", metadata)
    assert result["modifier_hallucination_count"] == expected_count


def test_traffic_light_without_ego_info(metric_objects):
    metric_objects(SceneObjectType.TRAFFIC_LIGHT, colors=[ColorModifier.RED])
    metadata = {"scene": [{"class": "traffic_light", "state": "Green"}]}

    # Without the ego signal state, the light's color is unverifiable.
    result = check_hallucinations_in_instruction("", metadata)
    assert result["modifier_hallucination_count"] == 0


@pytest.mark.parametrize('lane_type,kind,flag', [
    ('Driving', SceneObjectType.DRIVING_LANE, None),
    ('Parking', SceneObjectType.PARKING_LANE, 'parking'),
    ('Shoulder', SceneObjectType.SHOULDER, 'shoulder'),
    ('Sidewalk', SceneObjectType.SIDEWALK, 'sidewalk'),
    ('Biking', SceneObjectType.BIKE_LANE, 'bike_lane'),
])
@pytest.mark.parametrize('type_key', ['type', 'type:'])
def test_lane_evidence(metric_objects, lane_type, kind, flag, type_key):
    metric_objects(kind, positions=[PositionModifier.LEFT])
    info = {'class': 'ego_info', 'ego_lane': {'type': 'Driving'},
            'left_lanes': [{type_key: lane_type, 'width': 0.5}], 'right_lanes': []}
    if flag:
        info.update({f'{flag}_left': False, f'{flag}_right': False})
    assert check_hallucinations_in_instruction('', {'scene_objects': [info]})['object_hallucination_count'] == 0
    assert check_hallucinations_in_instruction('', {'scene_objects': [info]})['modifier_hallucination_count'] == 0
    info['right_lanes'], info['left_lanes'] = info['left_lanes'], []
    assert check_hallucinations_in_instruction('', {'scene': [info]})['modifier_hallucination_count'] == 1
    info['right_lanes'] = []
    assert check_hallucinations_in_instruction('', {'scene': [info]})['object_hallucination_count'] == (0 if kind == SceneObjectType.DRIVING_LANE else 1)


def test_lane_flags_and_unverifiable_features(metric_objects):
    metric_objects(SceneObjectType.SHOULDER, positions=[PositionModifier.FRONT])
    assert check_hallucinations_in_instruction('', {'scene': [{'class': 'ego_info', 'shoulder_left': True}]})['modifier_hallucination_count'] == 0
    assert check_hallucinations_in_instruction('', {'scene': [{'class': 'ego_info', 'ego_lane': {'type': 'Driving'}}]})['object_hallucination_count'] == 0
    for kind in [SceneObjectType.ROAD, SceneObjectType.JUNCTION, SceneObjectType.INTERSECTION]:
        metric_objects(kind)
        assert check_hallucinations_in_instruction('', {'scene': []})['object_hallucination_count'] == 0


def test_narrow_shoulders(metric_objects):
    metric_objects(SceneObjectType.SHOULDER, positions=[PositionModifier.LEFT])
    metadata = {"scene_objects": [{
        "class": "ego_info",
        "ego_lane": {"type:": "Driving", "width": 2.75},
        "left_lanes": [{"type:": "Shoulder", "width": 0.5}],
        "right_lanes": [],
        "shoulder_left": False,
        "shoulder_right": False,
    }]}

    # The collector's flags exclude narrow shoulders, but the lane list does not.
    result = check_hallucinations_in_instruction("", metadata)
    assert result["object_hallucination_count"] == 0
    assert result["modifier_hallucination_count"] == 0


def test_prepositional_turn_direction():
    doc = dependency_doc(['Turn', 'to', 'the', 'left'], [0, 0, 3, 1],
                         ['ROOT', 'prep', 'det', 'pobj'], ['VERB', 'ADP', 'DET', 'NOUN'],
                         ['turn', 'to', 'the', 'left'])
    assert parse_position_modifier(doc[3]) == PositionModifier.NONE


def test_end_to_end_metrics():
    result = check_hallucinations_in_instruction('Follow the red car ahead.', {
        'scene_objects': [{'class': 'car', 'color_name': 'Red', 'position': [10, 0, 0]}],
    })
    assert result == {'object_count': 1, 'modifier_count': 2,
                      'object_hallucination_count': 0, 'modifier_hallucination_count': 0}
    assert check_hallucinations_in_instruction('Accelerate to 50 km/h.', {'scene': []}) == {
        'object_count': 0, 'modifier_count': 0,
        'object_hallucination_count': 0, 'modifier_hallucination_count': 0,
    }


@pytest.mark.parametrize("instruction", ["Follow the red car ahead", "Follow the red car ahead."])
def test_red_car_behind_and_blue_car_ahead(instruction):
    metadata = {"scene_objects": [
        {"class": "car", "color_name": "red", "position": [-10, 0, 0]},
        {"class": "car", "color_name": "blue", "position": [10, 0, 0]},
    ]}

    # Each car contradicts one modifier. The mention must match one whole car,
    # rather than combining the red color of one with the position of the other.
    result = check_hallucinations_in_instruction(instruction, metadata)
    assert result == {
        "object_count": 1,
        "modifier_count": 2,
        "object_hallucination_count": 0,
        "modifier_hallucination_count": 1,
    }
