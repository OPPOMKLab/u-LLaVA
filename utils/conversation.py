"""
Conversation prompt template of uLLaVA
Adapted from: https://github.com/haotian-liu/LLaVA/blob/main/llava/conversation.py
"""

import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: Tuple[str, str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += message
                    else:
                        ret += " " + message + self.sep2
            else:
                    ret += ""
            ret = ret.lstrip(self.sep)
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=[
        ["Human", "Give three tips for staying healthy."],
        ["Assistant",
            "Sure, here are three tips for staying healthy:\n"
            "1. Exercise regularly: Regular physical activity can help improve your overall health and wellbeing. "
            "It can also help reduce your risk of chronic conditions such as obesity, diabetes, heart disease, "
            "and certain cancers. Aim for at least 150 minutes of moderate-intensity aerobic exercise or "
            "75 minutes of vigorous-intensity aerobic exercise per week, along with muscle-strengthening "
            "activities at least two days per week.\n"
            "2. Eat a balanced diet: Eating a balanced diet that is rich in fruits, "
            "vegetables, whole grains, lean proteins, and healthy fats can help support "
            "your overall health. Try to limit your intake of processed and high-sugar foods, "
            "and aim to drink plenty of water throughout the day.\n"
            "3. Get enough sleep: Getting enough quality sleep is essential for your physical "
            "and mental health. Adults should aim for seven to nine hours of sleep per night. "
            "Establish a regular sleep schedule and try to create a relaxing bedtime routine to "
            "help improve the quality of your sleep."]
    ],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


conv_v1_2 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=[
        ["Human", "What are the key differences between renewable and non-renewable energy sources?"],
        ["Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n"]
    ],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


conv_vicuna_v1_1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


conv_sep1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=[
        ["Human", "Hi!"],
        ["Assistant", "Hi there! How can I help you today?"]
    ],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


simple_conv_legacy = Conversation(
    system="You are designed to assist human with a variety of tasks using natural language."
           "Follow the instructions carefully.",
    roles=("Human", "Assistant"),
    messages=[
        ["Human", "Hi!\n\n### Response:"],
        ["Assistant", "Hi there!  How can I help you today?\n"]
    ],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


conv_simple = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


conv_sep2 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)


conversation_lib = {
    'conv_simple': conv_simple,
    'conv_sep2': conv_sep2,
    'conv_llama2': conv_llama2,
}

default_conversation = conv_sep2

if __name__ == "__main__":
    conv = conversation_lib['conv_simple'].copy()
    conv.append_message(conv.roles[0], 'Describe the image.')
    conv.append_message(conv.roles[1], 'Sure, it is a dog.')
    print(conv.messages)
    print(conv.get_prompt())
