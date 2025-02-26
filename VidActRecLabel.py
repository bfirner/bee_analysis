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
import cv2
import math
import numpy
import sys
import sdl2.events
import sdl2.ext
import sdl2.keyboard
import sdl2.mouse
import sdl2.rect
import sdl2.sdlttf
import os
import yaml

from PIL import Image

import utility.annotations
import utility.image_provider as ip


def getComponentRect(component):
    return sdl2.rect.SDL_Rect(component.x, component.y, component.x + component.size[0], component.y + component.size[1])


class AnnotatorUI():
    def __init__(self, annotations, sprite_factory, ui_factory, renderer, font, image_size):
        self.visible = True
        # When annotating, assume annotations apply to the same target as previously used.
        self.annotations = annotations
        ################
        # Object annotation
        self.last_object = None
        if 0 < len(self.objects):
            self.last_object = self.objects[0]
            self.setVisibility(True)
        self.bbox_sprites = []
        for _ in self.objects:
            self.bbox_sprites.append(None)
        self.temp_bbox = None
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
        # Patch selection
        self.patch_selection = False
        # For numeric inputs
        # TODO Use an area to display the number
        self.number_field = self.ui_factory.from_color(sdl2.ext.TEXTENTRY, (255, 255, 255, 192), size=(image_size[0]//3, 3 * self.font_height))
        self.number_field.position = (2*image_size[0]//3, image_size[1]//2)
        self.number_entry = False
        ################
        # Frame labelling
        self.labelling = False
        self.label_keep = False
        self.prompt_label_on = sdl2.ext.Texture(renderer, font.render_text("Labelling on."))
        self.prompt_label_off = sdl2.ext.Texture(renderer, font.render_text("Labelling off."))
        self.prompt_label_keep = sdl2.ext.Texture(renderer, font.render_text("Keep frame."))
        self.prompt_label_discard = sdl2.ext.Texture(renderer, font.render_text("Discard frame."))

    def toggleLabelling(self):
        self.labelling = not self.labelling

    def toggleKeep(self):
        self.label_keep = not self.label_keep

    def updateLabels(self, frames):
        """Update the frame labels according to self.labelling and self.label_keep."""
        if type(frames) is not list:
            frames = [frames]
        if self.labelling:
            for frame in frames:
                utility.annotations.setFrameLabel(self.annotations, frame, self.label_keep)

    @property
    def objects(self):
        """Getter for the objects in the annotations."""
        return list(self.annotations['objects'].keys())

    def getAnnotations(self):
        return self.annotations

    def cycleObjects(self):
        if self.last_object is not None:
            object_index = self.objects.index(self.last_object)
            self.last_object = self.objects[(object_index + 1) % len(self.objects)]

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
                self.bbox_sprites.append(None)
            self.text_entry = False
            sdl2.keyboard.SDL_StopTextInput()

    def beginNumericInput(self):
        if not self.number_entry:
            self.setVisibility(True)
            sdl2.keyboard.SDL_SetTextInputRect(getComponentRect(self.entry_field))
            sdl2.keyboard.SDL_StartTextInput()
            self.number_entry = True
            self.num_buffer = ""

    def endNumericInput(self):
        """End numeric input (started with beginNumericInput) and return the number, or return None without any entry."""
        number = None
        if self.number_entry:
            if 0 < len(self.num_buffer):
                number = int(self.num_buffer)
                self.num_buffer = None
            self.number_entry = False
            sdl2.keyboard.SDL_StopTextInput()
        return number

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
            # The first two points should be the upper left, but the user could have drawn the bbox
            # in the other direction.  Use min and max to get the correct ordering.
            bbox = [min(self.bb_begin[x], self.bb_end[x]) for x in range(2)] + [max(self.bb_begin[x], self.bb_end[x]) for x in range(2)]
            # Don't save if the size isn't at least a pixel in both dimensions
            if 0 < bbox[2] - bbox[0] and 0 < bbox[3] - bbox[1]:
                utility.annotations.addFrameAnnotation(self.annotations, self.last_object, self.bb_frame, "bbox", bbox)
        # End bounding box selection
        self.bb_begin = None
        self.bb_end = None
        self.bb_selection = False

    def beginPatch(self):
        """Begin selection of a patch area."""
        self.patch_selection = True
        # x, y origin
        self.patch_origin = None
        # width and height from the origin
        self.patch_dims = None

    def endPatch(self):
        """Return the patch coordinates."""
        if self.patch_selection and self.patch_origin is not None and self.patch_dims is not None:
            self.patch_selection = False
            # Change any negative dimensions into positive ones and adjust the origin as appropriate
            for dim in range(2):
                if self.patch_dims[dim] < 0:
                    self.patch_origin[dim] = self.patch_origin[dim] + self.patch_dims[dim]
                    self.patch_dims[dim] = self.patch_dims[dim] * -1
            patch_info = {
                'crop_x_offset': self.patch_origin[0],
                'crop_y_offset': self.patch_origin[1],
                'width': self.patch_dims[0],
                'height': self.patch_dims[1]
            }
            return patch_info
        self.patch_selection = False
        return None

    def handleEvent(self, event):
        """Handle the possible text input event. Return true if the event is handled here."""
        if self.text_entry and event.type == sdl2.events.SDL_TEXTINPUT:
            new_input = sdl2.ext.compat.stringify(event.text.text, "utf-8")
            self.name_buffer += new_input
            return True
        elif self.number_entry and event.type == sdl2.events.SDL_TEXTINPUT:
            new_input = sdl2.ext.compat.stringify(event.text.text, "utf-8")
            self.num_buffer += new_input
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
        elif self.patch_selection and event.type == sdl2.events.SDL_MOUSEBUTTONDOWN and event.button.button == sdl2.mouse.SDL_BUTTON_LEFT:
            # Initialize on the first click, adjust on other clicks
            if self.patch_origin is None:
                self.patch_origin = [event.button.x, event.button.y]
            # TODO Check where the click occurs and begin adjust just x or y and either the origin or the dimensions
            return True
        elif self.patch_selection and self.patch_origin is not None and event.type == sdl2.events.SDL_MOUSEMOTION:
            self.patch_dims = [event.button.x - self.patch_origin[0], event.button.y - self.patch_origin[1]]
            return True

        return False

    def active(self):
        return self.text_entry or self.bb_selection or self.patch_selection

    def deactivate(self):
        """Call when the user completes the current action (such as with the enter key)"""
        if self.text_entry:
            self.endNameInput()
        elif self.bb_selection:
            self.endSelector()
        elif self.patch_selection:
            self.endPatch()

    def hasBbox(self, objname, cur_frame):
        """True if the bounding box for the object exists at this frame."""
        return objname in self.objects and utility.annotations.hasFrameAnnotation(self.annotations, objname, cur_frame, 'bbox')

    def clearBbox(self, objname, cur_frame):
        utility.annotations.removeFrameAnnotation(self.annotations, objname, cur_frame, 'bbox')

    def getBbox(self, objname, cur_frame):
        """Get the bounding box for the given object.
        Returns:
            None or [left, top, right, bottom]
        """
        if objname not in self.objects:
            return None
        if not utility.annotations.hasFrameAnnotation(self.annotations, objname, cur_frame, 'bbox'):
            return None
        return utility.annotations.getFrameAnnotation(self.annotations, objname, cur_frame, 'bbox')

    def addBbox(self, objname, cur_frame, bbox):
        """Add the bounding box for the given object and frame.
        """
        if objname in self.objects:
            utility.annotations.addFrameAnnotation(self.annotations, objname, cur_frame, "bbox", bbox)

    def render(self, renderer, spriterenderer, cur_frame):
        if self.visible:
            sprite_targets = []
            ######
            # Upper left elements that show the current object
            # Begin with the text box
            if self.text_entry:
                renderer.copy(self.prompt_new_object, dstrect=(10, self.font_height//2))
                # Render the name of the ongoing object below the prompt
                cur_object = sdl2.ext.Texture(renderer, self.font.render_text("New object: {}".format(self.name_buffer)))
            else:
                renderer.copy(self.prompt_accept_object, dstrect=(10, self.font_height//2))
                # Render the name of the current object below the prompt
                cur_object = sdl2.ext.Texture(renderer, self.font.render_text("Current object: {}".format(self.last_object)))
            renderer.copy(cur_object, dstrect=(10, 3*self.font_height//2))
            # The texture is no longer required once rendered.
            # TODO We could check if things have changed and reuse it, but this is just text rendering.
            del cur_object
            ######
            # The bounding boxes
            if self.bb_selection and self.bb_begin is not None and self.bb_end is not None:
                # Something is being drawn, render it
                bsize = (abs(self.bb_end[0] - self.bb_begin[0]), abs(self.bb_end[1] - self.bb_begin[1]))
                if bsize[0] > 0 and bsize[1] > 0:
                    if self.temp_bbox is None:
                        self.temp_bbox = self.sprite_factory.from_color((50, 50, 200, 15), size=bsize)
                        # NOTE SDL seems to require these to use alpha transparency, even though it was already set in the from_color function.
                        sdl2.SDL_SetTextureBlendMode(self.temp_bbox.texture, sdl2.SDL_BLENDMODE_BLEND)
                        sdl2.SDL_SetTextureAlphaMod(self.temp_bbox.texture, 100)
                    else:
                        # TODO FIXME We actually just want to resize the bounding box, but sprites don't support that. This implies that we don't really want to be using a sprite.
                        #self.temp_bbox.size = bsize
                        del self.temp_bbox
                        self.temp_bbox = self.sprite_factory.from_color((50, 50, 200, 15), size=bsize)
                        # NOTE SDL seems to require these to use alpha transparency, even though it was already set in the from_color function.
                        sdl2.SDL_SetTextureBlendMode(self.temp_bbox.texture, sdl2.SDL_BLENDMODE_BLEND)
                        sdl2.SDL_SetTextureAlphaMod(self.temp_bbox.texture, 100)
                    self.temp_bbox.position = min(self.bb_begin[0], self.bb_end[0]), min(self.bb_begin[1], self.bb_end[1])
                    #renderer.copy(self.temp_bbox, dstrect=self.temp_bbox.position)
                    sprite_targets.append(self.temp_bbox)
            for objidx, objname in enumerate(self.objects):
                if utility.annotations.hasFrameAnnotation(self.annotations, objname, cur_frame, 'bbox'):
                    # Draw a box around the annotated object
                    bbox_coords = utility.annotations.getFrameAnnotation(self.annotations, objname, cur_frame, 'bbox')
                    bsize = (abs(bbox_coords[2] - bbox_coords[0]), abs(bbox_coords[3] - bbox_coords[1]))
                    if self.bbox_sprites[objidx] is None:
                        self.bbox_sprites[objidx] = self.sprite_factory.from_color((50, 200, 50, 15), size=bsize)
                        # NOTE SDL seems to require these to use alpha transparency, even though it was already set in the from_color function.
                        sdl2.SDL_SetTextureBlendMode(self.bbox_sprites[objidx].texture, sdl2.SDL_BLENDMODE_BLEND)
                        sdl2.SDL_SetTextureAlphaMod(self.bbox_sprites[objidx].texture, 100)
                    else:
                        # TODO FIXME We actually just want to resize the bounding box, but sprites don't support that. This implies that we don't really want to be using a sprite.
                        #self.bbox_sprites[objidx].size = bsize
                        tmp = self.bbox_sprites[objidx]
                        self.bbox_sprites[objidx] = self.sprite_factory.from_color((50, 200, 50, 15), size=bsize)
                        # NOTE SDL seems to require these to use alpha transparency, even though it was already set in the from_color function.
                        sdl2.SDL_SetTextureBlendMode(self.bbox_sprites[objidx].texture, sdl2.SDL_BLENDMODE_BLEND)
                        sdl2.SDL_SetTextureAlphaMod(self.bbox_sprites[objidx].texture, 100)
                        del tmp

                    self.bbox_sprites[objidx].position = bbox_coords[:2]
                    #renderer.copy(self.bbox_sprites[objidx], dstrect=self.bbox_sprites[objidx].position)
                    sprite_targets.append(self.bbox_sprites[objidx])
            ######
            # Patch box
            if self.patch_selection and self.patch_origin is not None and self.patch_dims is not None:
                # Something is being drawn, render it
                bsize = (abs(self.patch_dims[0]), abs(self.patch_dims[1]))
                if bsize[0] > 0 and bsize[1] > 0:
                    if self.temp_bbox is None:
                        self.temp_bbox = self.sprite_factory.from_color((200, 200, 50, 15), size=bsize)
                        # NOTE SDL seems to require these to use alpha transparency, even though it was already set in the from_color function.
                        sdl2.SDL_SetTextureBlendMode(self.temp_bbox.texture, sdl2.SDL_BLENDMODE_BLEND)
                        sdl2.SDL_SetTextureAlphaMod(self.temp_bbox.texture, 100)
                    else:
                        # TODO FIXME We actually just want to resize the bounding box, but sprites don't support that. This implies that we don't really want to be using a sprite.
                        #self.temp_bbox.size = bsize
                        del self.temp_bbox
                        self.temp_bbox = self.sprite_factory.from_color((200, 200, 50, 15), size=bsize)
                        # NOTE SDL seems to require these to use alpha transparency, even though it was already set in the from_color function.
                        sdl2.SDL_SetTextureBlendMode(self.temp_bbox.texture, sdl2.SDL_BLENDMODE_BLEND)
                        sdl2.SDL_SetTextureAlphaMod(self.temp_bbox.texture, 100)
                    self.temp_bbox.position = min(self.patch_origin[0], self.patch_origin[0] + self.patch_dims[0]), min(self.patch_origin[1], self.patch_origin[1] + self.patch_dims[1])
                    #renderer.copy(self.temp_bbox, dstrect=self.temp_bbox.position)
                    sprite_targets.append(self.temp_bbox)
            ######
            # The current label and labelling state
            if self.labelling:
                renderer.copy(self.prompt_label_on, dstrect=(10, 6*self.font_height//2))
            else:
                renderer.copy(self.prompt_label_off, dstrect=(10, 6*self.font_height//2))
            if utility.annotations.getFrameLabel(self.annotations, cur_frame):
                renderer.copy(self.prompt_label_keep, dstrect=(10, 9*self.font_height//2))
            else:
                renderer.copy(self.prompt_label_discard, dstrect=(10, 9*self.font_height//2))
            spriterenderer.render(sprite_targets)

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
    sprite_factory = sdl2.ext.SpriteFactory(sdl2.ext.TEXTURE, renderer=renderer)

    # TODO Testing this rendering approach
    spriterenderer = sprite_factory.create_sprite_render_system(window)

    # TODO FIXME Handle case when the default font is not present
    default_font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    if os.path.exists(default_font):
        font = sdl2.ext.FontTTF(default_font, '14px', (255, 255, 255, 255))
    font_height = sdl2.sdlttf.TTF_FontLineSkip(font.get_ttf_font())

    # For UI components
    uifactory = sdl2.ext.UIFactory(sprite_factory)

    # Make a UI element for annotations
    # It accepts new object names and allows the user to cycle through them.
    aui = AnnotatorUI(annotations, sprite_factory, uifactory, renderer, font, provider.imageSize())

    uiprocessor = sdl2.ext.UIProcessor()

    # Autotracker for easier human annotation
    tracker = None

    # For annotation interpolation
    interp_begin = None

    ################
    # Image display

    running = True
    frame_num = 0
    frame = provider.getFrame(frame_num)

    ################
    # Convert the image to an SDL surface then texture.
    def updateFrameUI(frame, renderer, frame_num, provider):
        # Try to avoid doing this more frequently than we load new frames as the underlying
        # SDL_ConvertSurface call inside of pillow_to_surface is memory intensive.
        img_surface = sdl2.ext.image.pillow_to_surface(Image.fromarray((frame*255).astype(numpy.uint8)))
        img_tx = sdl2.ext.Texture(renderer, img_surface)
        # The current state text
        txt_rendered = font.render_text("Frame {}/{}".format(frame_num, provider.totalFrames()))
        return img_surface, img_tx, txt_rendered
    img_surface, img_tx, txt_rendered = updateFrameUI(frame, renderer, frame_num, provider)


    while running:
        # Prepare the screen
        renderer.clear()

        ################
        # Autolabelling with an object tracker
        if tracker is not None:
            # NOTE We are leaving the image in RGB format, even though OpenCV likes BGR
            found, foundbbox = tracker.update((frame * 255).astype(numpy.uint8))
            if found:
                # Update the bounding box of the currently tracked object
                bbox = list(foundbbox[0:2]) + [foundbbox[0] + foundbbox[2], foundbbox[1] + foundbbox[3]]
                aui.addBbox(aui.last_object, frame_num, bbox)
            else:
                print("Tracked object lost.")
                del tracker
                tracker = None

        ################
        # Copy the current image to the render window
        renderer.copy(img_tx, dstrect=(0, 0))

        ################
        # Draw current state
        tx = sdl2.ext.Texture(renderer, txt_rendered)
        renderer.copy(tx, dstrect=(10, provider.imageSize()[1] - 2*font_height))

        #aui.render(renderer, frame_num)
        aui.render(renderer, spriterenderer, frame_num)

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
        if tracker is not None:
            next_frame = frame_num + 1
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
            elif sdl2.ext.key_pressed(events, 'c'):
                print("Clearing annotation.")
                aui.clearBbox(aui.last_object, frame_num)
            elif sdl2.ext.key_pressed(events, 'a'):
                print("Adding annotation.")
                # An annotation must apply to a target
                if 0 == len(aui.objects):
                    aui.beginNameInput()
                else:
                    # Begin the annotation by creating a selector
                    aui.beginSelector(frame_num)
            elif sdl2.ext.key_pressed(events, 't'):
                # Toggle the tracker auto annotation.
                if tracker is None and aui.hasBbox(aui.last_object, frame_num):
                    print("Enabling tracker.")
                    # Start tracking the aui.last_object, if there is such an object and it is visible on this frame.
                    # Vit tracker uses the example from the opencv zoo https://github.com/opencv/opencv_zoo
                    # The example uses the model in the same repository: https://github.com/opencv/opencv_zoo/blob/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx
                    if cv2.__version__.split('.')[:2] == ["4", "10"]:
                        params = cv2.TrackerVit_Params()
                        # TODO FIXME Put this into the SDL resources?
                        params.net = "../opencv_zoo/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx"
                        tracker = cv2.TrackerVit_create(params)
                    else:
                        # TODO FIXME Other versions also have the vit tracker, and there are other trackers as well.
                        raise RuntimeError("No supported tracker in installed version of open cv.")
                    bbox = aui.getBbox(aui.last_object, frame_num)
                    # Convert to x, y, width, height as open CV expects.
                    bbox = bbox[:2] + [bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    tracker.init((frame * 255).astype(numpy.uint8), bbox)
                    # Autoadvance to the next frame
                    next_frame = frame_num + 1
                else:
                    print("Disabling tracker.")
                    del tracker
                    tracker = None
            elif sdl2.ext.key_pressed(events, 'n'):
                print("Adding new object.")
                aui.beginNameInput()
            elif sdl2.ext.key_pressed(events, 'p'):
                print("Creating patch params.")
                if not aui.patch_selection:
                    aui.beginPatch()
                else:
                    patch_params = aui.endPatch()
                    if patch_params is not None:
                        patch_file = os.path.join(args.source_dir, "image_proc.yaml")
                        print(f"Saving patch parameters into {patch_file}")
                        with open(patch_file, "w", newline=None) as pfile:
                            yaml.dump(patch_params, pfile)
            elif sdl2.ext.key_pressed(events, 'j'):
                # Get frame input and then jump to a frame.
                aui.beginNumericInput()
            elif sdl2.ext.key_pressed(events, 'l'):
                # Toggle labelling enable
                aui.toggleLabelling()
            elif sdl2.ext.key_pressed(events, 'k'):
                # Toggle labelling type
                aui.toggleKeep()
                aui.updateLabels(frame_num)
            elif sdl2.ext.key_pressed(events, 'i'):
                # Interpolation begin or end
                if interp_begin is not None:
                    if aui.hasBbox(aui.last_object, frame_num):
                        # Perform interpolation
                        first_interp = min(interp_begin, frame_num)
                        last_interp = max(interp_begin, frame_num)
                        first_bbox = aui.getBbox(aui.last_object, first_interp)
                        last_bbox = aui.getBbox(aui.last_object, last_interp)
                        deltas = numpy.float32(last_bbox) - numpy.float32(first_bbox)
                        for interframe in range(first_interp+1, last_interp):
                            new_bbox = numpy.int32(numpy.round((numpy.int32(first_bbox) + (interframe - first_interp)/(last_interp - first_interp) * deltas))).tolist()

                            aui.addBbox(aui.last_object, interframe, new_bbox)
                    else:
                        print("No bounding box at {} for interpolation.".format(frame_num))
                    interp_begin = None
                else:
                    if aui.hasBbox(aui.last_object, frame_num):
                        interp_begin = frame_num
                        print("Interpolating from {}".format(interp_begin))
                    else:
                        print("No bounding box at {} for interpolation.".format(frame_num))
            elif sdl2.ext.key_pressed(events, sdl2.SDLK_RETURN):
                # Complete new name entry
                if aui.text_entry:
                    aui.endNameInput()
                if aui.number_entry:
                    target = aui.endNumericInput()
                    if target is not None:
                        print(f"Jumping to frame {target}")
                        next_frame = target
            elif sdl2.ext.key_pressed(events, sdl2.SDLK_BACKSPACE):
                # Remove a letter from the text entry
                if aui.text_entry:
                    aui.name_buffer = aui.name_buffer[:-1]
                if aui.number_entry:
                    aui.num_buffer = aui.num_buffer[:-1]
            elif sdl2.ext.key_pressed(events, sdl2.SDLK_TAB):
                if tracker is None:
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
        old_frame = frame_num
        if next_frame != frame_num and provider.hasFrame(next_frame):
            frame_num = next_frame
            frame = provider.getFrame(frame_num)
            # Manually mark sufaces and textures for deletion. These can overwhelm pythons very robust garbage collector.
            del img_surface
            del img_tx
            del txt_rendered
            # Create the image surface and texture for the renderer
            img_surface, img_tx, txt_rendered = updateFrameUI(frame, renderer, frame_num, provider)
        elif tracker is not None:
            # Stop tracker if there are no more frames.
            del tracker
            tracker = None
        # Handle any label updates as frames are advanced.
        if old_frame != frame_num:
            if abs(old_frame - frame_num) == 1:
                aui.updateLabels(frame_num)
            else:
                # This is a group of frames that should be updated
                sign = int(math.copysign(1, frame_num - old_frame))
                aui.updateLabels(list(range(old_frame+sign, frame_num+sign, sign)))

    # Clean up and quit
    font.close()
    renderer.destroy()
    window.close()
    sdl2.ext.quit()
    return 0


if __name__ == '__main__':
    sys.exit(main())
