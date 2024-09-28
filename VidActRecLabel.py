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
import sdl2.mouse
import sdl2.rect
import sdl2.sdlttf
import os

from PIL import Image

import utility.annotations
import utility.image_provider as ip


def getComponentRect(component):
    return sdl2.rect.SDL_Rect(component.x, component.y, component.x + component.size[0], component.y + component.size[1])


class AnnotatorUI():
    def __init__(self, annotations, sprite_factory, ui_factory, renderer, font, image_size):
        self.visible = False
        # When annotating, assume annotations apply to the same target as previously used.
        self.annotations = annotations
        self.last_object = None
        if 0 < len(self.objects):
            self.last_object = self.objects[0]
            self.setVisibility(True)
        # TODO Kind of ugly to be passed the factories here
        self.ui_factory = ui_factory
        self.sprite_factory = sprite_factory
        self.font = font
        self.image_size = image_size
        self.font_height = sdl2.sdlttf.TTF_FontLineSkip(font.get_ttf_font())
        # Entry field with 3x the font height to allow for both instructions and the user input
        self.entry_field = self.ui_factory.from_color(sdl2.ext.TEXTENTRY, (255, 255, 255, 192), size=(image_size[0]//3, 3 * self.font_height))
        self.entry_field.position = (10, 10)
        self.text_entry = False
        self.bb_selection = False
        self.prompt_new_object = sdl2.ext.Texture(renderer, font.render_text("Enter name of new object."))
        self.prompt_accept_object = sdl2.ext.Texture(renderer, font.render_text("Press tab to cycle objects."))

    @property
    def objects(self):
        """Getter for the objects in the annotations."""
        return list(self.annotations['objects'].keys())

    def getAnnotations(self):
        return self.annotations

    def cycleObjects(self):
        if self.last_object is not None:
            object_index = self.objects.index(self.last_object)
            self.last_object = self.objects[(self.last_object + 1) % len(self.objects)]

    def beginNameInput(self):
        if not self.text_entry:
            self.setVisibility(True)
            sdl2.keyboard.SDL_SetTextInputRect(getComponentRect(self.entry_field))
            sdl2.keyboard.SDL_StartTextInput()
            self.text_entry = True
            self.name_buffer = ""

    def endNameInput(self):
        if self.text_entry:
            if 0 < len(self.name_buffer) and self.name_buffer not in self.objects:
                utility.annotations.addObject(self.annotations, self.name_buffer)
                self.last_object = self.name_buffer
                self.name_buffer = None
            self.text_entry = False
            sdl2.keyboard.SDL_StopTextInput()

    def beginSelector(self, cur_frame):
        """Begin bounding box selection."""
        self.bb_selection = True
        # No bounding box defined yet
        self.bb_begin = None
        self.bb_end = None
        self.bb_frame = cur_frame

    def endSelector(self):
        """End bounding box selection and add the current bbox into the annotations."""
        # Make annotations for self.last_object from point self.bb_begin to self.bb_end
        if self.bb_begin is not None and self.bb_end is not None:
            # TODO FIXME No, the first two points should be the upper left, but the user could have draw the bbox in the other direction
            # The bounding box should have the upper left coordinate first, then the lower right
            bbox = [min(self.bb_begin[x], self.bb_end[x]) for x in range(2)] + [max(self.bb_begin[x], self.bb_end[x]) for x in range(2)]
            # Don't save if the size isn't at least a pixel in both dimensions
            if 0 < bbox[2] - bbox[0] and 0 < bbox[3] - bbox[1]:
                utility.annotations.addFrameAnnotation(self.annotations, self.last_object, self.bb_frame, "bbox", bbox)
        # End bounding box selection
        self.bb_begin = None
        self.bb_end = None
        self.bb_selection = False

    def handleEvent(self, event):
        """Handle the possible text input event. Return true if the event is handled here."""
        if self.text_entry and event.type == sdl2.events.SDL_TEXTINPUT:
            new_input =  sdl2.ext.compat.stringify(event.text.text, "utf-8")
            print("Got new input {}".format(new_input))
            self.name_buffer += new_input
            return True
        elif self.bb_selection and event.type == sdl2.events.SDL_MOUSEBUTTONDOWN and event.button.button == sdl2.mouse.SDL_BUTTON_LEFT:
            self.bb_begin = [event.button.x, event.button.y]
            return True
        elif self.bb_selection and self.bb_begin is not None and event.type == sdl2.events.SDL_MOUSEMOTION:
            # Drag the ending location
            self.bb_end = [event.button.x, event.button.y]
            return True
        elif self.bb_selection and event.type == sdl2.events.SDL_MOUSEBUTTONUP and event.button.button == sdl2.mouse.SDL_BUTTON_LEFT:
            # Drag the ending location
            self.bb_end = [event.button.x, event.button.y]
            # No more moving of the bounding box
            self.endSelector()
            return True
        return False

    def active(self):
        return self.text_entry or self.bb_selection

    def deactivate(self):
        """Call when the user completes the current action (such as with the enter key)"""
        if self.text_entry:
            self.endNameInput()
        elif self.bb_selection:
            self.endSelector()

    def render(self, renderer, cur_frame):
        if self.visible:
            # The text box
            if self.text_entry:
                renderer.copy(self.prompt_new_object, dstrect=(10, self.font_height//2))
                # Render the name of the ongoing object below the prompt
                cur_object = sdl2.ext.Texture(renderer, self.font.render_text("New object: {}".format(self.name_buffer)))
            else:
                renderer.copy(self.prompt_accept_object, dstrect=(10, self.font_height//2))
                # Render the name of the current object below the prompt
                cur_object = sdl2.ext.Texture(renderer, self.font.render_text("Current object: {}".format(self.last_object)))
            renderer.copy(cur_object, dstrect=(10, 3*self.font_height//2))
            # The bounding boxes
            if self.bb_selection and self.bb_begin is not None and self.bb_end is not None:
                # Something is being drawn, render it
                bsize = (abs(self.bb_end[0] - self.bb_begin[0]), abs(self.bb_end[1] - self.bb_begin[1]))
                if bsize[0] > 0 and bsize[1] > 0:
                    bbox = self.sprite_factory.from_color((50, 50, 200, 15), size=bsize)
                    bbox.position = min(self.bb_begin[0], self.bb_end[0]), min(self.bb_begin[1], self.bb_end[1])
                    # NOTE SDL seems to require these to use alpha transparency, even though it was already set in the from_color function.
                    sdl2.SDL_SetTextureBlendMode(bbox.texture, sdl2.SDL_BLENDMODE_BLEND)
                    sdl2.SDL_SetTextureAlphaMod(bbox.texture, 100)
                    renderer.copy(bbox, dstrect=bbox.position)
            for objname in self.objects:
                if utility.annotations.hasFrameAnnotation(self.annotations, objname, cur_frame, 'bbox'):
                    # Draw a box around the annotated object
                    bbox_coords = utility.annotations.getFrameAnnotation(self.annotations, objname, cur_frame, 'bbox')
                    bsize = (abs(bbox_coords[2] - bbox_coords[0]), abs(bbox_coords[3] - bbox_coords[1]))
                    bbox = self.sprite_factory.from_color((50, 200, 50, 15), size=bsize)
                    bbox.position = bbox_coords[:2]
                    # NOTE SDL seems to require these to use alpha transparency, even though it was already set in the from_color function.
                    sdl2.SDL_SetTextureBlendMode(bbox.texture, sdl2.SDL_BLENDMODE_BLEND)
                    sdl2.SDL_SetTextureAlphaMod(bbox.texture, 100)
                    renderer.copy(bbox, dstrect=bbox.position)

    def setVisibility(self, visible):
        self.visible = visible


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
        annotations = utility.annotations.initializeAnnotations(provider)

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

    # Make a UI element for annotations
    # It accepts new object names and allows the user to cycle through them.
    # TODO Add in bounding box drawing with an addBbox function
    aui = AnnotatorUI(annotations, factory, uifactory, renderer, font, provider.imageSize())

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

        aui.render(renderer, frame_num)

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
        handled = False
        for event in events:
            # See possible events in https://github.com/libsdl-org/SDL/blob/SDL2/include/SDL_events.h or https://github.com/py-sdl/py-sdl2/blob/master/sdl2/events.py
            if event.type == sdl2.SDL_QUIT:
                running = False
            # NOTE -- The if statement condition has a side effect
            if not aui.handleEvent(event):
                unhandled_events.append(event)
            else:
                handled = True
        events = unhandled_events
        # TODO Find a way to consume keyboard events properly in the aui
        # TODO Also handle the backspace key
        if handled:
            events = []
        # A key was pressed
        if sdl2.ext.key_pressed(events):
            # See https://github.com/py-sdl/py-sdl2/blob/0.9.16/sdl2/ext/input.py
            # TODO Use the scale key to abort annotation operations.
            # Handle a keyboard quit
            if sdl2.ext.key_pressed(events, 'q'):
                running = False
            elif sdl2.ext.key_pressed(events, 's'):
                print("Saving annotations.")
                utility.annotations.saveAnnotations(aui.getAnnotations(), annotation_file)
            elif sdl2.ext.key_pressed(events, 'a'):
                print("Adding annotation.")
                # An annotation must apply to a target
                if 0 == len(aui.objects):
                    aui.beginNameInput()
                else:
                    # Begin the annotation by creating a selector
                    aui.beginSelector(frame_num)
            elif sdl2.ext.key_pressed(events, 'n'):
                print("Adding new object.")
                aui.beginNameInput()
            elif sdl2.ext.key_pressed(events, sdl2.SDLK_RETURN):
                # Complete new name entry
                if aui.text_entry:
                    aui.endNameInput()
            elif sdl2.ext.key_pressed(events, sdl2.SDLK_BACKSPACE):
                # Remove a letter from the text entry
                if aui.text_entry:
                    aui.name_buffer = aui.name_buffer[:-1]
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
