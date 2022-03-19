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
    video = 'assets/petal_20220217_160339.mp4'
    logo = 'assets/datlacuache_white.png'

    object_tracking_in_video(video, logo_path=logo)


if __name__ == '__main__':
    main()