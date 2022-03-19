# ============================================================================
# Author: Rodolfo Ferro @ Datlacuache
# Twitter: @rodo_ferro
#
# Script: Trackable object.
#
# ABOUT COPYING OR USING PARTIAL INFORMATION:
# This script was originally created by Rodolfo Ferro. Any
# explicit usage of this script or its contents is granted
# according to the license provided and its conditions.
# ============================================================================


class TrackableObject:
    """Trackable Object."""

    def __init__(self, objectID, centroid):
        """Class constructor
        
        Parameters
        ----------
        object_id : int
            The ID of the object to be tracked.
        centroid : tuple
            The centroid of the object.
        """

        self.objectID = objectID
        self.centroids = [centroid]
        self.init_position = False
        self.counted = False