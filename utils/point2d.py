#  ================================================================
#  Created by Gregory Kramida on 9/17/18.
#  Copyright (c) 2018 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================


class Point2d:
    """ Point class represents and manipulates x,y coords. """

    def __init__(self, x=0.0, y=0.0, coordinates=None):
        """ Create a new point at the origin """
        if coordinates is not None:
            self.x = coordinates[0]
            self.y = coordinates[1]
        else:
            self.x = x
            self.y = y

    def __repr__(self):
        if self.x % 1.0 == 0 and self.y % 1.0 == 0:
            return "[{:d},{:d}]".format(int(self.x), int(self.y))
        return "[{:>03.2f},{:>03.2f}]".format(self.x, self.y)

    def __add__(self, other):
        return Point2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point2d(self.x - other.x, self.y - other.y)