#  ================================================================
#  Created by Gregory Kramida on 8/9/16.
#  Copyright (c) 2016 Gregory Kramida
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


class Argument(object):
    setting_file_location_wildcard = '!settings_file_location'

    def __init__(self, default,
                 nargs='?',
                 arg_type=str,
                 action='store',
                 arg_help="Documentation N/A",
                 console_only=False,
                 required=False,
                 shorthand=None,
                 setting_file_location=False):
        """
        @rtype: Argument
        @type name: str
        @param name: argument name -- to be used in both console and config file
        @type default: object
        @param default: the default value
        @type nargs: int | str
        @param nargs: number of arguments. See python documentation for ArgumentParser.add_argument.
        @type arg_type: type | str
        @param arg_type: type of value to expect during parsing
        @type action: str | function
        @param action: action to perform with the argument value during parsing
        @type arg_help: str
        @param arg_help: documentation for this argument
        @type console_only: bool
        @param console_only: whether the argument is for console use only or for both config file & console
        @type required: bool
        @param required: whether the argument is required
        @type shorthand: str
        @param shorthand: shorthand to use for argument in console
        @type setting_file_location: bool
        @param setting_file_location: whether to
        """
        self.default = default
        self.required = required
        self.console_only = console_only
        self.nargs = nargs
        self.type = arg_type
        self.action = action
        if setting_file_location:
            self.help = arg_help + ("| If set to '" + Argument.setting_file_location_wildcard + "' and a " +
                                    " settings file is provided, will be set to the location of the settings file.")
        else:
            self.help = arg_help
        self.setting_file_location = setting_file_location

        if shorthand is None:
            self.shorthand = None
        else:
            self.shorthand = "-" + shorthand
