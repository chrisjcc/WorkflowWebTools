#!/usr/bin/env python

"""
.. describe:: update_history.py

A python script for updating the history database.

Can take arguments that are file names or urls.
If a local file doesn't exist, the url is assumed.
The input file is then added to the central history
database, as defined in your server config file.
If no arguments are given, the central history
database is just generated as empty.

It is recommended to set up a cron job to update your
central history regularly.
For example adding the line::

    0 * * * * <path/to>/update_history.py <URL to errors>

to your crontab will automatically update the
central database every hour.
Duplicate entries will not be added.

:author: Daniel Abercrombie <dabercro@mit.edu>
"""

import os
import sys
import sqlite3

from WorkflowWebTools import errorutils
from WorkflowWebTools import serverconfig


def main(*args):
    """
    Updates the history database.

    :param args: list of error files to add to the history.
    """
    conn = sqlite3.connect(serverconfig.workflow_history_path())
    curs = conn.cursor()
    curs.execute('SELECT name FROM sqlite_master WHERE type="table" and name="workflows"')

    if not curs.fetchone():
        errorutils.create_table(curs)

    for arg in args:
        print "Adding " + arg
        errorutils.add_to_database(curs, arg)

    conn.commit()
    conn.close()


if __name__ == '__main__':
    serverconfig.LOCATION = os.path.dirname(os.path.realpath(__file__))
    main(*(sys.argv[1:]))
