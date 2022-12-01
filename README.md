# sd-promptbook

![promptbook](https://github.com/space-nuko/sd-promptbook/raw/master/static/promptbook.png)

This extension for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) allows you to save prompt snippets and combine them together into a "recipe" format for higher-level prompt exploration.

This is very useful for quickly visualizing the effects of a group of keywords on the result image, or injecting a character that requires a lot of keywords to properly reproduce into a complex prompt.

## Usage

### Creating prompt cards

First order of business is to generate some prompt cards, these are `.png` files that hold the prompt information for promptbook in the EXIF metadata as JSON. Each prompt card should contain the small snippets of the positive/negative prompts you want to combine together later.

In the `Scripts` dropdown for txt2img, select `Promptbook Generation` and then the `Save Prompt` tab. Now enter the prompt you want to save into the positive/negative fields of the txt2img UI and press `Generate` like usual. This will save a new prompt card to the output directory (default is `extensions/sd-promptbook/promptbook/prompts`)

The `Example prompt` fields are used when generating the prompt card's image. Typically I use `masterpiece, highest quality, highres, 1girl` in the positive prompt so there's at least a clearly defined female subject in the resulting images. `Append example prompts` means to place your snippet prompt before the example prompts.

Note that if you have a negative prompt you normally use for txt2img, it will get added to the snippet unless you remove it. Usually you want to keep the negative prompt blank when generating prompt cards, except for a single card that captures the negative prompt keywords you normally use. You would then add that negative prompt card to all your recipes later on.

You can also batch generate prompt cards using the `Batch file` field. If it's not blank, the file's contents will override whatever's in the txt2img prompt fields. This field should point to a `.txt` file containing one positive prompt per line, in this case the negative prompt is ignored.

Try generating a few more cards with some keywords you find useful. Alternatively you can find other people's prompt card `.png`s somewhere and put them into your `promptbook/prompts` folder. Here is an example that encodes `masterpiece, best quality, highres` in the positive prompt and some common negative prompt keywords, you can download and use it yourself:

![masterpiece + unwanted + bad anatomy](https://github.com/space-nuko/sd-promptbook/raw/master/static/masterpiece%20%2B%20unwanted%20%2B%20bad%20anatomy.png)

As you can see the card will have a before/after comparison so you can better understand how the keywords within will affect the output. Optionally you can disable the comparison image, for example if you're generating hundreds of prompts in a batch session with just one seed.

### Browsing the promptbook

Now with some prompt cards at hand, go to the `Promptbook` tab and browse through the pages. Here you can view your available prompt cards and combine them into recipes.

Note that if you want to edit one of your saved prompt cards, the way to do this is to send the prompt to `txt2img` using the button and generate another prompt card from it. The reason for this is to ensure the cover image remains accurate after you change the prompt keywords.

### Using recipes

![prompt_merge](https://github.com/space-nuko/sd-promptbook/raw/master/static/prompt_merge.png)

The interface at the bottom of the promptbook allows you to merge several of your prompt cards together into a "recipe" for use with the included `Promptbook Generation` script. To do this, select one and press the `Add prompt` button. It will add the prompt to the selected slot in sequential order. Alternatively you can click on one of the numbered buttons on the side of each slot to select which one to insert into.

Each individual prompt can have a custom strength set. It's quite useful for affecting the weights of a bunch of keywords at once. This feature uses the same prompt parser that the base webui uses.

If you're missing a prompt card for the text you want to add, you can use the `(Custom)` prompt type selectable through the dropdown on each row. This lets you enter a custom positive/negative prompt for that row.

Once you're satisfied with the prompt setup, press the `Generate Recipe` button. The JSON of the recipe will be output. Copy it and switch to txt2img, then select the `Promptbook Generation > Generate` tab and paste it into the `Request JSON` field there. Afterwards you can adjust your model and parameters like usual and click `Generate`. Each of the positive/negative prompts will be combined together in sequential order with any custom strengths applied. 

The usual txt2img prompt can also be appended or prepended to the result output by the recipe evaluator for quick testing.

### Viewing prompt results

![generated](https://github.com/space-nuko/sd-promptbook/raw/master/static/generated.png)

You can view the sample images resulting from adding a prompt to a recipe via the `generated` tab in the `Promptbook` section, so long as they aren't moved from their original location. First refresh the list of prompts by clicking the blue refresh button, then select the prompt you want to view images for. As you combine a lot of prompts this becomes more and more useful for gathering examples of outputs across all prompts, filtered down to a single subject or concept.
