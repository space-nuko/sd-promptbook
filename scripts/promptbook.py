import sys
sys.dont_write_bytecode = True

import os
import shutil
import time
import stat
import json
import math
import hashlib
import pdb

import gradio as gr
import modules.extras
import modules.ui
import modules.scripts as scripts
import modules.prompt_parser as prompt_parser
import pprint
import piexif
import piexif.helper

from functools import partial
from PIL import Image
from copy import copy
from modules.shared import opts, cmd_opts
from modules.processing import process_images, Processed
from modules.images import FilenameGenerator
from modules import shared, images, scripts, script_callbacks
from pathlib import Path
from typing import List, Tuple


##########################################################################################


PROMPTBOOK_VERSION = "1.0"

refresh_symbol = '\U0001f504'  # 🔄

def sanitize_prompt(prompt):
    return prompt.strip().strip(',').strip()


class Recipe:
    def __init__(self, parts):
        self.parts = parts

    def serialize(self):
        return {
            "version": PROMPTBOOK_VERSION,
            "parts": [part.serialize() if part else None for part in self.parts]
        }

    def deserialize(data):
        parts = [RecipePart.deserialize(part_json) if part_json else None for part_json in data["parts"]]

        return Recipe(parts)

    def to_json(self):
        return json.dumps(self.serialize())

    def from_json(jsdata):
        data = json.loads(jsdata)
        assert isinstance(data, dict)
        return Recipe.deserialize(data)

    def from_json_array(jsdata):
        data = json.loads(jsdata)
        assert isinstance(data, list)
        return [Recipe.deserialize(entry) for entry in data]


class RecipePart:
    def __init__(self, prompt, strength):
        self.prompt: Prompt = prompt
        self.strength = strength

    def serialize(self):
        return {
            "prompt": self.prompt.serialize(slim=True),
            "strength": self.strength
        }

    def deserialize(data):
        prompt = Prompt.deserialize(data["prompt"])
        strength = data["strength"]

        return RecipePart(prompt, strength)

    def to_json(self):
        return json.dumps(self.serialize())

    def from_json(jsdata):
        data = json.loads(jsdata)
        return Recipe.deserialize(data)


class Prompt:
    def __init__(self, name, description, positive, negative, sha256=None):
        self.name = name
        self.description = description
        self.positive = sanitize_prompt(positive)
        self.negative = sanitize_prompt(negative)

        if name == "(Custom)":
            self.sha256 = ""
        elif sha256 is None:
            m = hashlib.sha256()
            m.update(self.positive.encode())
            m.update(self.negative.encode())
            self.sha256 = m.hexdigest()
        else:
            self.sha256 = sha256

    def load_from_file(png_filepath):
        image = Image.open(png_filepath)
        return Prompt.load_from_image(image)

    def load_from_image(image):
        _, info = images.read_info_from_image(image)
        return Prompt.from_json(info["promptbook_prompt"])

    def serialize(self, slim=False):
        data = {
            "name": self.name,
            "description": self.description,
            "positive": self.positive,
            "negative": self.negative,
            "sha256": self.sha256,
        }

        if not slim:
            data["version"] = PROMPTBOOK_VERSION

        return data

    def deserialize(data):
        name = data["name"]
        description = data["description"]
        positive = data["positive"]
        negative = data["negative"]
        sha256 = data.get("sha256", None)

        return Prompt(name, description, positive, negative, sha256)

    def to_json(self):
        return json.dumps(self.serialize())

    def from_json(jsdata):
        data = json.loads(jsdata)
        return Prompt.deserialize(data)


####################################################################################################


SPECIAL_PROMPTS = ["(None)", "(Custom)"]


def get_prompt_path(prompt_name):
    return os.path.join(opts.promptbook_prompts_path, f"{prompt_name}.png")


class UiPromptMerge:
    def __init__(self, rows):
        self.rows = rows

    def prompt_choices_outputs(self):
        return [row.dropdown_prompt for row in self.rows]

    def add_prompt_outputs(self):
        return list(sum([(row.dropdown_prompt, row.slider_strength) for row in self.rows], ()))

    def select_button_outputs(self):
        return [row.select_button for row in self.rows]

    def output_prompt_outputs(self):
        return list(sum([(row.dropdown_prompt, row.slider_strength, row.custom_prompt, row.custom_negative_prompt) for row in self.rows], ()))

    @staticmethod
    def load_recipe(recipe_json):
        recipe = Recipe.from_json(recipe_json)
        results = ["(None)", 1.0, "", ""] * opts.promptbook_merge_max_rows

        for index, part in enumerate(recipe.parts):
            if part is None:
                continue
            elif part.prompt.name == "(Custom)":
                results[index * 4] = part.prompt.name
                results[index * 4 + 1] = part.strength
                results[index * 4 + 2] = part.prompt.positive
                results[index * 4 + 3] = part.prompt.negative
            else:
                prompt_path = get_prompt_path(part.prompt.name)
                if os.path.isfile(prompt_path):
                    loaded = Prompt.load_from_file(prompt_path)
                    if loaded.sha256 == part.prompt.sha256:
                        results[index * 4] = part.prompt.name
                        results[index * 4 + 1] = part.strength

        return results

    @staticmethod
    def output_recipe(*rest):
        recipe = UiPromptMerge.make_recipe(rest)
        return recipe.to_json()

    @staticmethod
    def clear_recipe():
        return ["(None)", 1.0, "", ""] * opts.promptbook_merge_max_rows

    @staticmethod
    def make_recipe(allvars):
        pairwise = list(zip(allvars[::4], allvars[1::4], allvars[2::4], allvars[3::4]))
        parts = []
        for pair in pairwise:
            prompt_name, strength, custom_prompt, custom_negative_prompt = pair
            if prompt_name != "" and prompt_name != "(None)":
                parts.append(UiPromptMergeRow.make_recipe_part(prompt_name, strength, custom_prompt, custom_negative_prompt))
            else:
                parts.append(None)

        while parts and parts[-1] is None:
            parts.pop()

        return Recipe(parts)


class UiPromptMergeRow:
    def __init__(self, index, row, delete_button, select_button, image_preview, dropdown_prompt, slider_strength, custom_prompt, custom_negative_prompt):
        self.index = index
        self.row = row
        self.delete_button = delete_button
        self.select_button = select_button
        self.image_preview = image_preview
        self.dropdown_prompt = dropdown_prompt
        self.slider_strength = slider_strength
        self.custom_prompt = custom_prompt
        self.custom_negative_prompt = custom_negative_prompt

    @staticmethod
    def make_recipe_part(prompt_name, strength, custom_prompt, custom_negative_prompt):
        if prompt_name == "(Custom)":
            prompt = Prompt("(Custom)", "", custom_prompt, custom_negative_prompt)
        else:
            prompt = Prompt.load_from_file(get_prompt_path(prompt_name))
        return RecipePart(prompt, strength)



def update_row(prompt_name):
    image_file = os.path.join(opts.promptbook_prompts_path, f"{prompt_name}.png")
    if prompt_name in SPECIAL_PROMPTS or not os.path.isfile(image_file):
        image_file = None

    is_custom = prompt_name == "(Custom)"
    return image_file, gr.Row.update(visible=is_custom)


def ui_prompt_merge():
    all_prompt_choices = SPECIAL_PROMPTS
    all_rows = []

    with gr.Row():
        with gr.Column():
            for i in range(0, opts.promptbook_merge_max_rows):
                with gr.Row():
                    with gr.Column():
                        with gr.Row(variant="panel") as row:
                            with gr.Column(scale=1):
                                with gr.Row():
                                    delete_button = gr.Button("❌", elem_id=f"promptbook_prompt_delete_button_{i}")
                                    select_button = gr.Button(f"{i}", elem_id=f"promptbook_prompt_select_button_{i}", variant="primary" if i == 0 else "secondary")
                            with gr.Column(scale=19):
                                with gr.Row():
                                    image_preview = gr.Image(elem_id=f"promptbook_prompt_preview_{i}", type="pil", show_label=False, interactive=False).style(width=64, height=64)
                                    dropdown_prompt = gr.Dropdown(choices=all_prompt_choices, interactive=True, label="Prompt")
                                    slider_strength = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, value=1.0, label="Strength", interactive=True)
                                with gr.Row(visible=False) as custom_row:
                                    custom_prompt = gr.components.Textbox(label="Custom prompt", lines=2)
                                    custom_negative_prompt = gr.components.Textbox(label="Custom negative prompt", lines=2)

                            all_rows.append(UiPromptMergeRow(i, row, delete_button, select_button, image_preview, dropdown_prompt, slider_strength, custom_prompt, custom_negative_prompt))

                dropdown_prompt.change(fn=update_row, inputs=[dropdown_prompt], outputs=[image_preview, custom_row])
                delete_button.click(fn=lambda: ("(None)", 1.0), inputs=None, outputs=[dropdown_prompt, slider_strength])
    with gr.Row():
        create_recipe_button = gr.Button("Generate Recipe", variant="primary")
        load_recipe_button = gr.Button("Load Recipe")
        clear_recipe_button = gr.Button("Clear Recipe")
    with gr.Row():
        recipe_json = gr.Textbox(label="Recipe Output")

    prompt_merge_ui = UiPromptMerge(all_rows)
    create_recipe_button.click(fn=UiPromptMerge.output_recipe, inputs=prompt_merge_ui.output_prompt_outputs(), outputs=[recipe_json])
    load_recipe_button.click(fn=UiPromptMerge.load_recipe, inputs=[recipe_json], outputs=prompt_merge_ui.output_prompt_outputs())
    clear_recipe_button.click(fn=UiPromptMerge.clear_recipe, inputs=None, outputs=prompt_merge_ui.output_prompt_outputs())

    return prompt_merge_ui


####################################################################################################


tabs_list = ["promptbook", "generated"]
num_of_prompts_per_page = 0
PROMPTS_PATH = os.path.join(scripts.basedir(), "promptbook/prompts")
GENERATED_PATH = os.path.join(scripts.basedir(), "promptbook/generated")


def traverse_all_files(curr_path, prompt_list) -> List[Tuple[str, os.stat_result]]:
    f_list = [(os.path.join(curr_path, entry.name), entry.stat()) for entry in os.scandir(curr_path)]
    for f_info in f_list:
        fname, fstat = f_info
        if os.path.splitext(fname)[1] == ".png":
            prompt_list.append(f_info)
        elif stat.S_ISDIR(fstat.st_mode):
            prompt_list = traverse_all_files(fname, prompt_list)
    return prompt_list


def get_all_prompts(sort_by, keyword):
    fileinfos = traverse_all_files(shared.opts.promptbook_prompts_path, [])
    keyword = keyword.strip(" ")
    if len(keyword) != 0:
        fileinfos = [x for x in fileinfos if keyword.lower() in x[0].lower()]
    if sort_by == "date":
        fileinfos = sorted(fileinfos, key=lambda x: -x[1].st_mtime)
    elif sort_by == "path name":
        fileinfos = sorted(fileinfos)

    filenames = [finfo[0] for finfo in fileinfos]
    return filenames


def get_prompt_page(page_index, filenames, keyword, sort_by):
    if page_index == 1 or page_index == 0 or len(filenames) == 0:
        filenames = get_all_prompts(sort_by, keyword)
    page_index = int(page_index)
    length = len(filenames)
    max_page_index = length // num_of_prompts_per_page + 1
    page_index = max_page_index if page_index == -1 else page_index
    page_index = 1 if page_index < 1 else page_index
    page_index = max_page_index if page_index > max_page_index else page_index
    idx_frm = (page_index - 1) * num_of_prompts_per_page
    prompt_list = filenames[idx_frm:idx_frm + num_of_prompts_per_page]

    visible_num = num_of_prompts_per_page if idx_frm + num_of_prompts_per_page < length else length % num_of_prompts_per_page
    visible_num = num_of_prompts_per_page if visible_num == 0 else visible_num

    load_info = "<div style='color:#999' align='center'>"
    load_info += f"{length} images in this directory, divided into {int((length + 1) // num_of_prompts_per_page  + 1)} pages"
    load_info += "</div>"
    return filenames, page_index, prompt_list, visible_num, ""


def show_prompt_info(tabname_box, num, page_index, filenames):
    file = filenames[int(num) + int((page_index - 1) * num_of_prompts_per_page)]
    tm = "<div style='color:#999' align='right'>" + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(os.path.getmtime(file))) + "</div>"
    return file, tm, num, file, ""


def populate_prompt_info(image):
    prompt = Prompt.load_from_image(image)
    return prompt.name, prompt.description, prompt.positive, prompt.negative, "", ""


def open_promptbook():
    tabname = "promptbook"
    with gr.Tab(tabname):
        with gr.Blocks(analytics_enabled=False):
            with gr.Row():
                first_page = gr.Button('First Page')
                prev_page = gr.Button('Prev Page')
                page_index = gr.Number(value=1, label="Page Index")
                next_page = gr.Button('Next Page')
                end_page = gr.Button('Last Page')
            with gr.Row(elem_id=tabname+'_promptbook'):
                with gr.Row():
                    with gr.Column(scale=3):
                        prompt_gallery = gr.Gallery(show_label=False, elem_id=tabname + "_promptbook_gallery").style(grid=6)
                        with gr.Row():
                            add_prompt_button = gr.Button("Add Prompt", elem_id=tabname + "_promptbook_add_prompt_btn")
                            add_prompt_index = gr.Number(label="Prompt Index", value=0)
                        prompt_merge_ui = ui_prompt_merge()
                    with gr.Column():
                        with gr.Row():
                            sort_by = gr.Radio(value="date", choices=["path name", "date"], label="sort by")
                            keyword = gr.Textbox(value="", label="keyword")
                        with gr.Row():
                            with gr.Column():
                                prompt_name = gr.Textbox(value="", label="Prompt Name", interactive=False)
                                prompt_description = gr.Textbox(value="", label="Prompt Description", interactive=False)
                                prompt_positive = gr.Textbox(label="Positive Prompt", interactive=False, lines=6)
                                prompt_negative = gr.Textbox(label="Negative Prompt", interactive=False, lines=6)
                                prompt_file_name = gr.Textbox(value="", label="File Name", interactive=False)
                                prompt_file_time = gr.HTML()
                            with gr.Row():
                                try:
                                    send_to_buttons = modules.generation_parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])
                                except:
                                    pass

                    # hiden items
                    with gr.Row(visible=False):
                        renew_page = gr.Button("Renew Page", elem_id=tabname+"_promptbook_renew_page")
                        visible_img_num = gr.Number()
                        tabname_box = gr.Textbox(tabname)
                        prompt_index = gr.Textbox(value=-1)
                        set_index = gr.Button('set_index', elem_id=tabname+"_promptbook_set_index")
                        filenames = gr.State([])
                        prompt_info_switch = gr.Image(type="pil")
                        info1 = gr.Textbox()
                        info2 = gr.Textbox()
                        load_switch = gr.Textbox(value="load_switch", label="load_switch")
                        turn_page_switch = gr.Number(value=1, label="turn_page_switch")
                        prompt_file_info_raw = gr.Textbox(label="Generate Info", interactive=False, lines=6)

    #turn page
    first_page.click(lambda s:(1, -s) , inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    next_page.click(lambda p, s: (p + 1, -s), inputs=[page_index, turn_page_switch], outputs=[page_index, turn_page_switch])
    prev_page.click(lambda p, s: (p - 1, -s), inputs=[page_index, turn_page_switch], outputs=[page_index, turn_page_switch])
    end_page.click(lambda s: (-1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    load_switch.change(lambda s:(1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    keyword.submit(lambda s:(1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    sort_by.change(lambda s:(1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    page_index.submit(lambda s: -s, inputs=[turn_page_switch], outputs=[turn_page_switch])
    renew_page.click(lambda s: -s, inputs=[turn_page_switch], outputs=[turn_page_switch])

    turn_page_switch.change(
        fn=get_prompt_page,
        inputs=[page_index, filenames, keyword, sort_by],
        outputs=[filenames, page_index, prompt_gallery, visible_img_num, prompt_file_info_raw]
    )
    turn_page_switch.change(fn=None, inputs=[tabname_box], outputs=None, _js="promptbook_turnpage")

    def reload_prompts():
        prompts = get_all_prompts("", "")
        choices = SPECIAL_PROMPTS + [os.path.splitext(os.path.basename(prompt))[0] for prompt in prompts]
        return [gr.update(choices=choices) for i in range(len(prompt_merge_ui.prompt_choices_outputs()))]
    turn_page_switch.change(fn=reload_prompts, inputs=[], outputs=prompt_merge_ui.prompt_choices_outputs())

    set_index.click(show_prompt_info,
                    _js="promptbook_get_current_img",
                    inputs=[
                        tabname_box,
                        prompt_index,
                        page_index,
                        filenames
                    ],
                    outputs=[
                        prompt_file_name,
                        prompt_file_time,
                        prompt_index,
                        prompt_info_switch,
                        prompt_file_info_raw
                    ])
    prompt_info_switch.change(fn=populate_prompt_info, inputs=[prompt_info_switch], outputs=[prompt_name, prompt_description, prompt_positive, prompt_negative])
    prompt_info_switch.change(fn=modules.extras.run_pnginfo, inputs=[prompt_info_switch], outputs=[info1, prompt_file_info_raw, info2])

    try:
        modules.generation_parameters_copypaste.bind_buttons(send_to_buttons, prompt_file_name, prompt_file_info_raw)
    except:
        pass

    def add_prompt(add_prompt_name, add_index, *rest):
        pairwise = list(zip(rest[::2], rest[1::2]))
        result = []

        for index, pair in enumerate(pairwise):
            prompt_name, strength = pair
            if index == add_index:
                print("Add " + add_prompt_name)
                prompt_name = add_prompt_name
                strength = 1.0
            result += [prompt_name, strength]

        add_index += 1
        if add_index >= len(pairwise):
            add_index = 0

        return [add_index] + result

    add_prompt_vars = prompt_merge_ui.add_prompt_outputs()
    add_prompt_button.click(fn=add_prompt, inputs=[prompt_name, add_prompt_index] + add_prompt_vars, outputs=[add_prompt_index] + add_prompt_vars)

    def update_select_buttons(selected_index):
        return [gr.Button.update(variant="primary" if selected_index == i else "secondary") for i in range(0, opts.promptbook_merge_max_rows)]
    add_prompt_index.change(fn=update_select_buttons, inputs=[add_prompt_index], outputs=prompt_merge_ui.select_button_outputs())

    for row in prompt_merge_ui.rows:
        row.select_button.click(fn=lambda i=row.index: i, inputs=None, outputs=[add_prompt_index])
        row.delete_button.click(fn=lambda i=row.index: i, inputs=None, outputs=[add_prompt_index])


def get_all_generated(prompt_name, sort_by, keyword):
    path = os.path.join(opts.promptbook_generated_path, prompt_name + ".json")
    if not os.path.isfile(path):
        return []

    with open(path, "r") as file:
        jsdata = json.load(file)

    # TODO use correct sha256sum
    results = []
    for files in jsdata["files"].values():
        for entry in files:
            if os.path.isfile(entry["filename"]):
                results.append(entry["filename"])

    return results


def get_generated_page(prompt_name, page_index, filenames, keyword, sort_by):
    if page_index == 1 or page_index == 0 or len(filenames) == 0:
        filenames = get_all_generated(prompt_name, sort_by, keyword)
    page_index = int(page_index)
    length = len(filenames)
    max_page_index = length // num_of_prompts_per_page + 1
    page_index = max_page_index if page_index == -1 else page_index
    page_index = 1 if page_index < 1 else page_index
    page_index = max_page_index if page_index > max_page_index else page_index
    idx_frm = (page_index - 1) * num_of_prompts_per_page
    prompt_list = filenames[idx_frm:idx_frm + num_of_prompts_per_page]

    visible_num = num_of_prompts_per_page if idx_frm + num_of_prompts_per_page < length else length % num_of_prompts_per_page
    visible_num = num_of_prompts_per_page if visible_num == 0 else visible_num

    load_info = "<div style='color:#999' align='center'>"
    load_info += f"{length} images in this directory, divided into {int((length + 1) // num_of_prompts_per_page  + 1)} pages"
    load_info += "</div>"
    return filenames, page_index, prompt_list, visible_num, ""


def show_generated_info(tabname_box, num, page_index, filenames):
    file = filenames[int(num) + int((page_index - 1) * num_of_prompts_per_page)]
    tm = "<div style='color:#999' align='right'>" + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(os.path.getmtime(file))) + "</div>"
    return file, tm, num, file, ""


all_prompts = []


def reload_prompts():
    global all_prompts
    filenames = get_all_prompts("", "")
    all_prompts = [os.path.splitext(os.path.basename(filename))[0] for filename in filenames]


def open_generated():
    tabname = "generated"
    with gr.Tab(tabname):
        with gr.Blocks(analytics_enabled=False):
            with gr.Row():
                first_page = gr.Button('First Page')
                prev_page = gr.Button('Prev Page')
                page_index = gr.Number(value=1, label="Page Index")
                next_page = gr.Button('Next Page')
                end_page = gr.Button('Last Page')
            with gr.Row(elem_id=tabname+'_promptbook'):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            prompt_name = gr.Dropdown(label='Prompt', elem_id="promptbook_prompt_name", choices=[])
                            modules.ui.create_refresh_button(prompt_name, reload_prompts, lambda: {"choices": sorted([x for x in all_prompts])}, "refresh_prompts")
                        seed_gallery = gr.Gallery(show_label=False, elem_id=tabname + "_promptbook_gallery").style(grid=6)

                    with gr.Column():
                        with gr.Row():
                            sort_by = gr.Radio(value="date", choices=["path name", "date"], label="sort by")
                            keyword = gr.Textbox(value="", label="keyword")
                        with gr.Row():
                            with gr.Column():
                                generated_file_info = gr.Textbox(label="Generate Info", interactive=False, lines=6)
                                generated_file_name = gr.Textbox(value="", label="File Name", interactive=False)
                                generated_file_time = gr.HTML()
                        with gr.Row(elem_id=tabname + "_promptbook_button_panel"):
                            try:
                                send_to_buttons = modules.generation_parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])
                            except:
                                pass

                    # hiden items
                    with gr.Row(visible=False):
                        renew_page = gr.Button("Renew Page", elem_id=tabname+"_promptbook_renew_page")
                        visible_img_num = gr.Number()
                        tabname_box = gr.Textbox(tabname)
                        generated_index = gr.Textbox(value=-1)
                        set_index = gr.Button('set_index', elem_id=tabname+"_promptbook_set_index")
                        filenames = gr.State([])
                        generated_info_switch = gr.Image(type="pil")
                        info1 = gr.Textbox()
                        info2 = gr.Textbox()
                        load_switch = gr.Textbox(value="load_switch", label="load_switch")
                        turn_page_switch = gr.Number(value=1, label="turn_page_switch")

    #turn page
    first_page.click(lambda s:(1, -s) , inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    next_page.click(lambda p, s: (p + 1, -s), inputs=[page_index, turn_page_switch], outputs=[page_index, turn_page_switch])
    prev_page.click(lambda p, s: (p - 1, -s), inputs=[page_index, turn_page_switch], outputs=[page_index, turn_page_switch])
    end_page.click(lambda s: (-1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    load_switch.change(lambda s:(1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    keyword.submit(lambda s:(1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    sort_by.change(lambda s:(1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    page_index.submit(lambda s: -s, inputs=[turn_page_switch], outputs=[turn_page_switch])
    renew_page.click(lambda s: -s, inputs=[turn_page_switch], outputs=[turn_page_switch])
    prompt_name.change(lambda s: -s, inputs=[turn_page_switch], outputs=[turn_page_switch])

    turn_page_switch.change(
        fn=get_generated_page,
        inputs=[prompt_name, page_index, filenames, keyword, sort_by],
        outputs=[filenames, page_index, seed_gallery, visible_img_num, generated_file_info]
    )
    turn_page_switch.change(fn=None, inputs=[tabname_box], outputs=None, _js="promptbook_turnpage")

    set_index.click(show_generated_info,
                    _js="promptbook_get_current_img",
                    inputs=[
                        tabname_box,
                        generated_index,
                        page_index,
                        filenames
                    ],
                    outputs=[
                        generated_file_name,
                        generated_file_time,
                        generated_index,
                        generated_info_switch,
                        generated_file_info
                    ])
    generated_info_switch.change(fn=modules.extras.run_pnginfo, inputs=[generated_info_switch], outputs=[info1, generated_file_info, info2])

    try:
        modules.generation_parameters_copypaste.bind_buttons(send_to_buttons, generated_file_name, generated_file_info)
    except:
        pass


TAB_INDEX_GENERATE = 0
TAB_INDEX_SAVE_PROMPT = 1


def merge_processed(processed_list, include_lone_images=True, rows=None, make_grid=True) -> Processed:
    # Temporary list of all the images that are generated to be populated into the grid.
    # Will be filled with empty images for any individual step that fails to process properly
    image_cache = []

    processed_result = None
    for processed in processed_list:
        try:
            for processed_image in processed.images:
                if processed_result is None:
                    # Use our first valid processed result as a template container to hold our full results
                    processed_result = copy(processed)
                    cell_mode = processed_image.mode
                    cell_size = processed_image.size
                    processed_result.images = []
                    if make_grid:
                        processed_result.images.append(Image.new(cell_mode, cell_size))

                image_cache.append(processed_image)
                if include_lone_images:
                    processed_result.images.append(processed_image)
                    processed_result.all_prompts.append(processed.prompt)
                    processed_result.all_seeds.append(processed.seed)
                    processed_result.infotexts.append(processed.infotexts[0])
        except Exception:
            image_cache.append(Image.new(cell_mode, cell_size))

    if not processed_result:
        raise RuntimeError("Unexpected error: draw_xy_grid failed to return even a single processed image")

    if make_grid:
        if rows is None:
            rows = math.ceil(math.sqrt(len(image_cache)))

        # TODO: layout vertically depending on image size
        grid = images.image_grid(image_cache, rows=rows)

        processed_result.images[0] = grid

    return processed_result


####################################################################################################


def join_prompts(p):
    return sanitize_prompt(", ".join([sanitize_prompt(prompt) for prompt in p]))


def apply_prompt_strength(prompt, strength):
    if strength == 1.0 or not prompt.strip():
        return prompt

    parsed = prompt_parser.parse_prompt_attention(prompt)
    for pair in parsed:
        pair[1] *= strength

    return "".join(["(" + pair[0] + ":" + ("%0.3f" % pair[1]) + ")" for pair in parsed])


def apply_recipe(p, recipe, existing_prompt_action):
    prompt_parts = []
    negative_prompt_parts = []

    for part in recipe.parts:
        if part is None:
            continue

        prompt = part.prompt

        positive = apply_prompt_strength(prompt.positive, part.strength)
        negative = apply_prompt_strength(prompt.negative, part.strength)

        if positive.strip():
            prompt_parts.append(positive)
        if negative.strip():
            negative_prompt_parts.append(negative)

    if p.prompt.strip():
        if existing_prompt_action == "Append":
            prompt_parts.append(p.prompt)
        elif existing_prompt_action == "Prepend":
            prompt_parts = [p.prompt] + prompt_parts

    if p.negative_prompt.strip():
        if existing_prompt_action == "Append":
            negative_prompt_parts.append(p.negative_prompt)
        elif existing_prompt_action == "Prepend":
            negative_prompt_parts = [p.negative_prompt] + negative_prompt_parts

    p.prompt = join_prompts(prompt_parts)
    p.negative_prompt = join_prompts(negative_prompt_parts)

    return process_images(p)


# HACK: Better way of getting result filename from a Processed (before script returns)?
def get_resulting_sample_filename(p, processed, seed, prompt, image, image_count, index):
    # Copied from modules/images.py
    namegen = FilenameGenerator(p, seed, prompt, image)
    basename = ""
    extension = opts.samples_format
    path = p.outpath_samples

    save_to_dirs = opts.save_to_dirs

    if save_to_dirs:
        dirname = namegen.apply(opts.directories_filename_pattern or "[prompt_words]").lstrip(' ').rstrip('\\ /')
        path = os.path.join(path, dirname)

    os.makedirs(path, exist_ok=True)

    if seed is None:
        file_decoration = ""
    elif opts.save_to_dirs:
        file_decoration = opts.samples_filename_pattern or "[seed]"
    else:
        file_decoration = opts.samples_filename_pattern or "[seed]-[prompt_spaces]"

    add_number = opts.save_images_add_number or file_decoration == ''

    if file_decoration != "" and add_number:
        file_decoration = "-" + file_decoration

    file_decoration = namegen.apply(file_decoration)

    if add_number:
        basecount = images.get_next_sequence_number(path, basename) - image_count
        fn = f"{basecount + index:05}"
        fullfn = os.path.join(path, f"{fn}{file_decoration}.{extension}")
    else:
        fullfn = os.path.join(path, f"{file_decoration}.{extension}")

    return fullfn


def append_generated(prompt, strength, output_img_filename):
    if prompt.name == "(Custom)" or prompt.sha256 == "":
        return

    path = os.path.join(opts.promptbook_generated_path, prompt.name + ".json")
    if os.path.isfile(path):
        with open(path, "r") as file:
            jsdata = json.load(file)
    else:
        jsdata = {
            "version": PROMPTBOOK_VERSION,
            "files": {}
        }

    if prompt.sha256 not in jsdata["files"]:
        jsdata["files"][prompt.sha256] = []

    jsdata["files"][prompt.sha256].append({"filename": output_img_filename, "strength": strength})

    with open(path, "w+") as file:
        json.dump(jsdata, file)


class Script(scripts.Script):
    def title(self):
        return "Promptbook Generation"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        with gr.Tabs():
            with gr.TabItem('Generate') as tab_generate:
                recipe_json = gr.Textbox(label="Recipe JSON")
                dropdown_txt2img_prompt = gr.Dropdown(choices=["Ignore", "Prepend", "Append"], value="Ignore", label="txt2img prompt")
                checkbox_save_grid = gr.Checkbox(label="Save grid", value=True)

            with gr.TabItem('Save Prompt') as tab_save:
                prompt_name = gr.Textbox(label="Prompt name (Leave blank to positive prompt as name)", value="")
                prompt_description = gr.Textbox(label="Prompt description", value="")
                overwrite_existing = gr.Checkbox(label="Overwrite existing", value=False)
                example_prompt = gr.Textbox(label="Example prompt", value="", lines=2)
                example_negative_prompt = gr.Textbox(label="Example negative prompt", value="", lines=2)
                append_example_prompts = gr.Checkbox(label="Append example prompts", value=False)
                batch_filename = gr.Textbox(label="Batch file (one positive prompt per line)", value="")
                batch_reset_seed = gr.Checkbox(label="Randomize seed each batch iteration", value=False)

        with gr.Row(visible=False):
            tab_index = gr.Number(value=0)

        tab_generate.select(lambda: (TAB_INDEX_GENERATE), inputs=None, outputs=[tab_index])
        tab_save.select(
            lambda p, pn: (
                TAB_INDEX_SAVE_PROMPT,
                shared.opts.promptbook_default_prompt if not p.strip() else p,
                shared.opts.promptbook_default_negative_prompt if not pn.strip() else pn),
            inputs=[example_prompt, example_negative_prompt],
            outputs=[tab_index, example_prompt, example_negative_prompt])

        return [
            tab_index,
            recipe_json,
            dropdown_txt2img_prompt,
            checkbox_save_grid,
            prompt_name,
            prompt_description,
            overwrite_existing,
            example_prompt,
            example_negative_prompt,
            append_example_prompts,
            batch_filename,
            batch_reset_seed
        ]

    def run(self,
            p,
            tab_index,
            gen_recipe_json,
            gen_txt2img_prompt,
            gen_save_grid,
            save_prompt_name,
            save_prompt_description,
            save_overwrite_existing,
            save_example_prompt,
            save_example_negative_prompt,
            save_append_example_prompts,
            save_batch_filename,
            save_batch_reset_seed):
        modules.processing.fix_seed(p)

        os.makedirs(opts.promptbook_generated_path, exist_ok=True)

        if tab_index == TAB_INDEX_GENERATE:
            return self.generate(p, gen_recipe_json, gen_txt2img_prompt, gen_save_grid)
        elif tab_index == TAB_INDEX_SAVE_PROMPT:
            return self.save_prompt(
                p,
                save_prompt_name,
                save_prompt_description,
                save_overwrite_existing,
                save_example_prompt,
                save_example_negative_prompt,
                save_append_example_prompts,
                save_batch_filename,
                save_batch_reset_seed
            )

    def generate(self, p, recipe_json, txt2img_prompt, save_grid):
        if recipe_json.strip().startswith("["):
            recipes = Recipe.from_json_array(recipe_json)
        else:
            recipes = [Recipe.from_json(recipe_json)]

        results = []

        for recipe in recipes:
            processed = apply_recipe(p, recipe, txt2img_prompt)
            results.append(processed)

            unwanted_grid_because_of_img_count = len(processed.images) < 2 and opts.grid_only_if_multiple
            has_grid = (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count

            for i in range(0, len(processed.all_seeds)):
                filename = get_resulting_sample_filename(p,
                                                         processed,
                                                         processed.all_seeds[i],
                                                         processed.all_prompts[i],
                                                         # ignore grid image
                                                         processed.images[i + 1 if has_grid else 0],
                                                         len(processed.all_seeds),
                                                         i)
                for part in recipe.parts:
                    if part is None:
                        continue
                    append_generated(part.prompt, part.strength, filename)

            if save_grid:
                images.save_image(processed.images[0], p.outpath_grids, "promptbook", prompt=p.prompt, seed=processed.seed, grid=True, p=p)

        if not results:
            raise Exception("No recipes succesfully generated images.")

        merged = merge_processed(results, include_lone_images=True, make_grid=False)

        return merged

    def save_prompt(self,
                    p,
                    prompt_name,
                    prompt_description,
                    overwrite_existing,
                    example_prompt,
                    example_negative_prompt,
                    append_example_prompts,
                    batch_filename,
                    batch_reset_seed):
        if prompt_name == "":
            prompt_name = p.prompt
        sanitized_name = images.sanitize_filename_part(prompt_name, replace_spaces=False)
        if not sanitized_name.strip():
            raise RuntimeError("Prompt name was blank.")

        if batch_filename != "" and not overwrite_existing:
            outpath = os.path.join(opts.promptbook_prompts_path, f"{sanitized_name}.png")
            if os.path.exists(outpath):
                raise RuntimeError(f"Prompt file already exists at {outpath}.")

        p.batch_count = 1
        p.batch_size = 1

        if batch_filename != "":
            with open(batch_filename, "r", encoding="utf-8") as file:
                prompts = [(line.strip(), "") for line in file]
        else:
            prompts = [(p.prompt, p.negative_prompt)]

        shared.total_tqdm.updateTotal(p.steps * p.n_iter * 2 * len(prompts))
        shared.state.job_count = p.n_iter * 2 * len(prompts)

        results = []

        for i, prompt in enumerate(prompts):
            if batch_reset_seed:
                p.seed = -1
                p.subseed = -1
                modules.processing.fix_seed(p)

            pos, neg = prompt

            filename = sanitized_name
            if batch_filename != "":
                filename += "_" + str(i) + "_" + images.sanitize_filename_part(pos, replace_spaces=False)
                outpath = os.path.join(opts.promptbook_prompts_path, f"{filename}.png")
                if os.path.exists(outpath) and not overwrite_existing:
                    print("File already exists, skipping: " + filename)
                    continue

            print(f"Generating prompt card '{prompt_name}'")
            print(f"Positive:{pos}")
            if neg != "":
                print(f"Negative:{neg}")

            shared.state.job = f"Prompt before ({i*2} of {shared.state.job_count})"
            p1 = copy(p)
            p1.prompt = example_prompt
            p1.negative_prompt = example_negative_prompt
            before: Processed = process_images(p1)

            shared.state.job = f"Prompt after ({i*2+1} of {shared.state.job_count})"
            p2 = copy(p)
            if append_example_prompts:
                p2.prompt = join_prompts([pos, example_prompt])
                p2.negative_prompt = join_prompts([neg, example_negative_prompt])
            else:
                p2.prompt = join_prompts([example_prompt, pos])
                p2.negative_prompt = join_prompts([example_negative_prompt, neg])
            after: Processed = process_images(p2)

            processed = merge_processed([before, after], include_lone_images=True, rows=1)

            original_prompt = after.infotexts[0]
            prompt = Prompt(filename, prompt_description, pos, neg)
            grid_outpath = images.save_image(
                processed.images[0],
                opts.promptbook_prompts_path,
                basename="",
                prompt=p2.prompt,
                seed=processed.seed,
                grid=True,
                p=p2,
                forced_filename=filename,
                # don't save the info .txt alongside like if info=<...> were passed
                # these images are mostly for comparison instead of generating something pretty
                existing_info={"promptbook_prompt": prompt.to_json(), "parameters": original_prompt},
                pnginfo_section_name=""
            )

            # sample_filename = get_resulting_sample_filename(p2, after, p2.all_seeds[0], processed.all_prompts[0], processed.images[0], 2, 1)
            # append_generated(prompt, 1.0, sample_filename)
            # append_generated(prompt, 1.0, grid_outpath)

            results.append(processed)

        if not results:
            raise Exception("No promptcards generated! Do all the files in the batch exist already?")

        return merge_processed(results, include_lone_images=True, make_grid=False)


def create_promptbook(pb):
    with gr.Tabs(elem_id="promptbook_tab"):
        open_promptbook()
        open_generated()


def on_ui_tabs():
    global num_of_prompts_per_page
    num_of_prompts_per_page = int(opts.promptbook_page_columns * opts.promptbook_page_rows)
    with gr.Blocks(analytics_enabled=False) as pb:
        create_promptbook(pb)
        gr.Checkbox(opts.promptbook_preload, elem_id="promptbook_preload", visible=False)
        gr.Textbox(",".join(tabs_list), elem_id="promptbook_tabnames_list", visible=False)
    return (pb, "Promptbook", "promptbook"),


DEFAULT_PROMPT = """masterpiece, high quality, highres, 1girl"""
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_MAX_ROWS = 15


def on_ui_settings():
    section = ('promptbook', "Promptbook")
    shared.opts.add_option("promptbook_preload", shared.OptionInfo(False, "Preload images at startup", section=section))
    shared.opts.add_option("promptbook_page_columns", shared.OptionInfo(6, "Number of columns on the page", section=section))
    shared.opts.add_option("promptbook_page_rows", shared.OptionInfo(6, "Number of rows on the page", section=section))
    shared.opts.add_option("promptbook_default_prompt", shared.OptionInfo(DEFAULT_PROMPT, "Default prompt to use when generating prompt covers", section=section))
    shared.opts.add_option("promptbook_default_negative_prompt", shared.OptionInfo(DEFAULT_NEGATIVE_PROMPT, "Default negative prompt to use when generating prompt covers", section=section))
    shared.opts.add_option("promptbook_prompts_path", shared.OptionInfo(PROMPTS_PATH, "Path containing prompt .png files", section=section))
    shared.opts.add_option("promptbook_generated_path", shared.OptionInfo(GENERATED_PATH, "Path containing files for generation stats", section=section))
    shared.opts.add_option("promptbook_merge_max_rows", shared.OptionInfo(DEFAULT_MAX_ROWS, "Maximum rows for the prompt merger.", section=section))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
