# ============================================================================
# Author: Rodolfo Ferro @ Datlacuache
# Twitter: @rodo_ferro
#
# Script: Video detection script.
#
# ABOUT COPYING OR USING PARTIAL INFORMATION:
# This script was originally created by Rodolfo Ferro. Any
# explicit usage of this script or its contents is granted
# according to the license provided and its conditions.
# ============================================================================

from ml.detect import detect_in_video


def main():
    """Main function."""

    video = 'assets/test.mp4'
    logo = 'assets/datlacuache_white.png'
    detect_in_video(video,
                    tracked_classes=['person'],
                    logo_path=logo,
                    position='bottom')


if __name__ == '__main__':
    main()