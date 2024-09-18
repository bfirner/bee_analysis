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
import sdl2.ext
import sdl2.keyboard
import sdl2.sdlttf
import os

from PIL import Image

import utility.image_provider as ip


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
    # UI Initialization
    sdl2.ext.init()

    window = sdl2.ext.Window("Hello world!", size=provider.imageSize())
    window.show()

    # TODO Verify that hardware acceleration should be used
    #renderflags = sdl2.SDL_RENDERER_SOFTWARE
    renderflags = (sdl2.SDL_RENDERER_ACCELERATED | sdl2.SDL_RENDERER_PRESENTVSYNC)
    renderer = sdl2.ext.Renderer(window, flags=renderflags)

    # TODO FIXME Handle case when the default font is not present
    default_font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    if os.path.exists(default_font):
        font = sdl2.ext.FontTTF(default_font, '14px', (255, 255, 255, 255))
    font_height = sdl2.sdlttf.TTF_FontLineSkip(font.get_ttf_font())

    ################
    # Image display

    running = True
    frame_num = 0
    frame = provider.getFrame(frame_num)

    while running:
        renderer.clear()

        # Convert the image to an SDL surface, then texture, then copy it to the render window
        img_surface = sdl2.ext.image.pillow_to_surface(Image.fromarray((frame*255).astype(numpy.uint8)))
        img_tx = sdl2.ext.Texture(renderer, img_surface)
        renderer.copy(img_tx, dstrect=(0, 0))

        # Draw current state
        txt_rendered = font.render_text("Frame {}/{}".format(frame_num, provider.totalFrames(), width=provider.imageSize()[0]/4))
        tx = sdl2.ext.Texture(renderer, txt_rendered)
        renderer.copy(tx, dstrect=(10, provider.imageSize()[1] - 2*font_height))

        # Redraw the window
        renderer.present()

        ################
        # Event handling
        events = sdl2.ext.get_events()
        running = True
        # Default to the same frame, but the user could advance or decrease the position
        next_frame = frame_num
        for event in events:
            # See possible events in https://github.com/libsdl-org/SDL/blob/SDL2/include/SDL_events.h or https://github.com/py-sdl/py-sdl2/blob/master/sdl2/events.py
            if event.type == sdl2.SDL_QUIT:
                running = False
        # A key was pressed
        if sdl2.ext.key_pressed(events):
            # Handle a keyboard quit
            if sdl2.ext.key_pressed(events, 'q'):
                running = False
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

    sdl2.ext.quit()
    return 0


if __name__ == '__main__':
    sys.exit(main())
