#! /usr/bin/python3

"""
Copyright © 2024 Bernhard Firner

Released under the MIT license as part of https://github.com/bfirner/bee_analysis
See https://github.com/bfirner/bee_analysis/blob/main/LICENSE for more details.

Training annotation tool for videos or image collections.
Per frame/image annotations will be stored in a annotations.yaml file alongside the data.

Requires pysdl2: pip install -U py-sdl2 or sudo apt-get install python3-sdl2 (depending upon your package versions)
NOTE: This is written for sdl2 verion 0.9.16
"""

import argparse
import numpy
import sys
import sdl2.events
import sdl2.ext
import sdl2.keyboard
import sdl2.rect
import sdl2.sdlttf
import os

from PIL import Image

import utility.annotations
import utility.image_provider as ip


def getComponentRect(component):
    return sdl2.rect.SDL_Rect(component.x, component.y, component.x + component.size[0], component.y + component.size[1])


class AnnotatorUI():
    def __init__(self, factory, renderer, font, image_size, known_objects):
        # When annotating, assume annotations apply to the same target as previously used.
        self.last_object = None
        self.objects = []
        self.factory = factory
        self.font = font
        self.image_size = image_size
        self.font_height = sdl2.sdlttf.TTF_FontLineSkip(font.get_ttf_font())
        # Entry field with 3x the font height to allow for both instructions and the user input
        self.entry_field = factory.from_color(sdl2.ext.TEXTENTRY, (255, 255, 255, 192), size=(image_size[0]//3, 3 * self.font_height))
        self.entry_field.position = (10, 10)
        #self.entry_field.input += lambda (entry, event): self.onInput(entry, event)
        #self.entry_field.editing += lambda (entry, event): self.inEdit(entry, event)
        self.text_entry = False
        self.prompt_new_object = sdl2.ext.Texture(renderer, font.render_text("Enter name of new object."))
        self.prompt_accept_object = sdl2.ext.Texture(renderer, font.render_text("Press tab to cycle objects."))
        self.visible = False


    def cycleObjects(self):
        if self.last_object is not None:
            self.last_object = (self.last_object + 1) % len(self.objects)


    def beginNameInput(self):
        if not self.text_entry:
            self.setVisibility(True)
            sdl2.keyboard.SDL_SetTextInputRect(getComponentRect(self.entry_field))
            sdl2.keyboard.SDL_StartTextInput()
            self.text_entry = True
            self.name_buffer = ""


    def endNameInput(self):
        if self.text_entry:
            if 0 < len(self.name_buffer):
                self.objects.append(self.name_buffer)
                self.name_buffer = None
                self.last_object = -1
            self.text_entry = False
            sdl2.keyboard.SDL_StopTextInput()


    def handleEvent(self, event):
        """Handle the possible text input event. Return true if the event is handled here."""
        if event.type == sdl2.events.SDL_TEXTINPUT:
            new_input =  sdl2.ext.compat.stringify(event.text.text, "utf-8")
            print("Got new input {}".format(new_input))
            self.name_buffer += new_input
            return True
        return False


    def onInput(self, entry, event):
        print("input text is now {}".format(entry.text))


    def onEdit(self, entry, event):
        print("edit text is now {}".format(entry.edit.text))

    def render(self, renderer):
        if self.visible:
            if self.text_entry:
                renderer.copy(self.prompt_new_object, dstrect=(10, self.font_height//2))
                # Render the name of the ongoing object below the prompt
                cur_object = sdl2.ext.Texture(renderer, self.font.render_text("New object: {}".format(self.name_buffer)))
            else:
                renderer.copy(self.prompt_accept_object, dstrect=(10, self.font_height//2))
                # Render the name of the current object below the prompt
                cur_object = sdl2.ext.Texture(renderer, self.font.render_text("Current object: {}".format(self.objects[self.last_object])))
            renderer.copy(cur_object, dstrect=(10, 3*self.font_height//2))


    def setVisibility(self, visible):
        self.visible = visible


    #def makeEntry(self, event):
    #    if self.text_entry is None:
    #        self.text_entry = factory.from_color((0, 0, 0, 128), size=(image_size[0]//2, 2 * self.font_height))


def main():
    parser = argparse.ArgumentParser(description='Metadata label creation tool.')

    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        help='Directory with data. Also the location where labels.yaml will be stored.')
    parser.add_argument(
        '--image_pattern',
        type=str,
        required=True,
        help="Image pattern. E.g. 'image_{:05d}.png' or 'video.mp4'")
    parser.add_argument(
        '--time_file',
        type=str,
        required=False,
        default="timestamps.csv",
        help="File with frame timestamps. Expecting columns with time_ns and frame_number.")

    args = parser.parse_args()

    ################
    # Get the image data
    video_path = os.path.join(args.source_dir, args.image_pattern)
    provider = ip.getImageProvider(video_path)

    ################
    # Load annotations.yaml

    annotation_file = os.path.join(args.source_dir, "annotations.yaml")
    annotations = utility.annotations.getAnnotations(annotation_file)
    if annotations is None:
        # Create some default annotations for this file, which will just be some empty lists
        annotations = {
            # No objects yet
            'objects': [],
            # No annotations yet, but prepare a table for each frame
            'frame_annotations': [{} for _ in range(provider.totalFrames())],
        }

    ################
    # UI Initialization
    sdl2.ext.init()

    window = sdl2.ext.Window("VidActRecLabel: {}".format(args.source_dir), size=provider.imageSize())
    window.show()

    # TODO Verify that hardware acceleration should be used
    #renderflags = sdl2.SDL_RENDERER_SOFTWARE
    renderflags = (sdl2.SDL_RENDERER_ACCELERATED | sdl2.SDL_RENDERER_PRESENTVSYNC)
    renderer = sdl2.ext.Renderer(window, flags=renderflags)
    factory = sdl2.ext.SpriteFactory(sdl2.ext.TEXTURE, renderer=renderer)

    # TODO FIXME Handle case when the default font is not present
    default_font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    if os.path.exists(default_font):
        font = sdl2.ext.FontTTF(default_font, '14px', (255, 255, 255, 255))
    font_height = sdl2.sdlttf.TTF_FontLineSkip(font.get_ttf_font())

    # For UI components
    uifactory = sdl2.ext.UIFactory(factory)

    # TODO Make an object named entry selector.
    # TODO It should display a name for the user to accept, allow tabbing through existing names, and entering a new name
    # TODO It adjusts the last_target variable (and should perhaps own that variable)
    aui = AnnotatorUI(uifactory, renderer, font, provider.imageSize(), annotations['objects'])

    uiprocessor = sdl2.ext.UIProcessor()

    ################
    # Image display

    running = True
    frame_num = 0
    frame = provider.getFrame(frame_num)

    while running:
        # Prepare the screen
        renderer.clear()

        ################
        # Convert the image to an SDL surface, then texture, then copy it to the render window
        img_surface = sdl2.ext.image.pillow_to_surface(Image.fromarray((frame*255).astype(numpy.uint8)))
        img_tx = sdl2.ext.Texture(renderer, img_surface)
        renderer.copy(img_tx, dstrect=(0, 0))

        ################
        # Draw current state
        txt_rendered = font.render_text("Frame {}/{}".format(frame_num, provider.totalFrames()))
        tx = sdl2.ext.Texture(renderer, txt_rendered)
        renderer.copy(tx, dstrect=(10, provider.imageSize()[1] - 2*font_height))

        aui.render(renderer)

        ################
        # TODO Draw annotations for this frame

        # Redraw the window
        renderer.present()

        ################
        # Event handling
        events = sdl2.ext.get_events()
        running = True
        # Default to the same frame, but the user could advance or decrease the position
        next_frame = frame_num
        # The annotator UI may consume the keyboard event
        unhandled_events = []
        for event in events:
            # See possible events in https://github.com/libsdl-org/SDL/blob/SDL2/include/SDL_events.h or https://github.com/py-sdl/py-sdl2/blob/master/sdl2/events.py
            if event.type == sdl2.SDL_QUIT:
                running = False
            if not aui.text_entry or not aui.handleEvent(event):
                unhandled_events.append(event)
        events = unhandled_events
        # A key was pressed
        if sdl2.ext.key_pressed(events):
            # See https://github.com/py-sdl/py-sdl2/blob/0.9.16/sdl2/ext/input.py
            # Handle a keyboard quit
            if sdl2.ext.key_pressed(events, 'q'):
                running = False
            elif sdl2.ext.key_pressed(events, 's'):
                print("Saving annotations.")
                utility.annotations.saveAnnotations(annotations, annotation_file)
            elif sdl2.ext.key_pressed(events, 'a'):
                print("Adding annotation.")
                # First we need to see which target this annotation applies to
                # TODO Prompt to create an annotation if none exists or select one if it does
            elif sdl2.ext.key_pressed(events, 'n'):
                print("Adding new object.")
                aui.beginNameInput()
            elif sdl2.ext.key_pressed(events, sdl2.SDLK_RETURN):
                # Complete new name entry
                if aui.text_entry:
                    aui.endNameInput()
            elif sdl2.ext.key_pressed(events, sdl2.SDLK_TAB):
                # Cycle through the known object types
                aui.cycleObjects()
            elif sdl2.ext.key_pressed(events, 'left'):
                next_frame = frame_num - 1
            elif sdl2.ext.key_pressed(events, 'right'):
                next_frame = frame_num + 1
            elif sdl2.ext.key_pressed(events, 'down'):
                next_frame = frame_num - 10
            elif sdl2.ext.key_pressed(events, 'up'):
                next_frame = frame_num + 10
        if not running:
            print("User quit.")
            break
        if next_frame != frame_num and provider.hasFrame(next_frame):
            frame_num = next_frame
            frame = provider.getFrame(frame_num)

    # Clean up and quit
    font.close()
    renderer.destroy()
    window.close()
    sdl2.ext.quit()
    return 0


if __name__ == '__main__':
    sys.exit(main())
