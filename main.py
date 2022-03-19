# ============================================================================
# Author: Rodolfo Ferro @ Datlacuache
# Twitter: @rodo_ferro
#
# Script: Video tracking script.
#
# ABOUT COPYING OR USING PARTIAL INFORMATION:
# This script was originally created by Rodolfo Ferro. Any
# explicit usage of this script or its contents is granted
# according to the license provided and its conditions.
# ============================================================================

from ml.track import object_tracking_in_video


def main():
    video = 'assets/clip.mp4'
    logo = 'assets/datlacuache_black.png'

    object_tracking_in_video(
        video,
        tracked_classes=['car', 'truck'],
        threshold=0.25,
        logo_path=logo,
        position='bottom'
    )


if __name__ == '__main__':
    main()