#!/usr/bin/env python2.7

# pylint: disable=wrong-import-position, no-self-use, invalid-name

"""
workflowtools.py
----------------

Script to run the WorkflowWebTools server.

:author: Daniel Abercrombie <dabercro@mit.edu>
"""

import os
import sys
import glob
import time
import datetime
import sqlite3

import cherrypy
from mako.lookup import TemplateLookup

from WorkflowWebTools import serverconfig

if __name__ == '__main__' or 'mod_wsgi' in sys.modules.keys():
    serverconfig.LOCATION = os.path.dirname(os.path.realpath(__file__))

from WorkflowWebTools import manageusers
from WorkflowWebTools import manageactions
from WorkflowWebTools import showlog
from WorkflowWebTools import listpage
from WorkflowWebTools import globalerrors
from WorkflowWebTools import clusterworkflows
from WorkflowWebTools import classifyerrors

from CMSToolBox import sitereadiness

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'templates')

GET_TEMPLATE = TemplateLookup(directories=[TEMPLATES_DIR],
                              module_directory=os.path.join(TEMPLATES_DIR, 'mako_modules')
                             ).get_template
"""Function to get templates from the relative ``templates`` directory"""


class WorkflowTools(object):
    """This class holds all of the exposed methods for the Workflow Webpage"""

    def __init__(self):
        """Initializes the service by creating clusters, if running webpage"""
        if __name__ == '__main__':
            self.cluster()

    @cherrypy.expose
    def index(self):
        """
        :returns: The welcome page
        :rtype: str
        """
        return GET_TEMPLATE('welcome.html').render()

    @cherrypy.expose
    def cluster(self):
        """
        The function is only accessible to someone with a verified account.

        Navigating to ``https://localhost:8080/cluster``
        causes the server to regenerate the clusters that it has stored.
        This is useful when the history database of past errors has been
        updated with relevant errors since the server has been started or
        this function has been called.

        :returns: a confirmation page
        :rtype: str
        """
        self.clusterer = clusterworkflows.get_clusterer(
            serverconfig.workflow_history_path(),
            serverconfig.all_errors_path())
        return GET_TEMPLATE('complete.html').render()

    @cherrypy.expose
    def showlog(self, search='', module='', limit=50):
        """
        This page, located at ``https://localhost:8080/showlog``,
        returns logs that are stored in an elastic search server.
        More details can be found at :ref:`elastic-search-ref`.
        If directed here from :ref:`workflow-view-ref`, then
        the search will be for the relevant workflow.

        :param str search: The search string
        :param str module: The module to look at, if only interested in one
        :param int limit: The limit of number of logs to show on a single page
        :returns: the logs from elastic search
        :rtype: str
        """
        logdata = showlog.give_logs(search, module, int(limit))
        if isinstance(logdata, dict):
            return GET_TEMPLATE('showlog.html').render(logdata=logdata,
                                                       search=search,
                                                       module=module,
                                                       limit=limit)

        return logdata

    @cherrypy.expose
    def globalerror(self, pievar='errorcode'):
        """
        This page, located at ``https://localhost:8080/globalerror``,
        attempts to give an overall view of the errors that occurred
        in each workflow at different sites.
        The resulting view is a table of piecharts.
        The rows and columns can be adjusted to contain two of the following:

        - Workflow step name
        - Site where error occurred
        - Exit code of the error

        The third variable is used to split the pie charts.
        This variable inside the pie charts can be quickly changed
        by submitting the form in the upper left corner of the page.

        The size of the piecharts depend on the total number of errors in a given cell.
        Each cell also has a tooltip, giving the total number of errors in the piechart.
        The colors of the piecharts show the splitting based on the ``pievar``.
        Clicking on the pie chart will show the splitting explicitly
        using the :ref:`list-wfs-ref` page.

        If the steps make up the rows,
        the default view will show you the errors for each campaign.
        Clicking on the campaign name will cause the rows to expand
        to show the original workflow and all ACDCs (whether or not the ACDCs have errors).
        Following the link of the workflow will bring you to :ref:`workflow-view-ref`.
        Clicking anywhere else in the workflow box
        will cause it to expand to show errors for each step.

        :param str pievar: The variable that the pie charts are split into.
                           Valid values are:

                           - errorcode
                           - sitename
                           - stepname

        :returns: the global views of errors
        :rtype: str
        """

        # For some reasons, we occasionally have to refresh this global errors page

        errors = globalerrors.get_errors(pievar, cherrypy.session)
        if pievar != 'stepname':

            # This pulls out the timestamp from the workflow parameters
            timestamp = lambda wkf: time.mktime(
                datetime.datetime(
                    *(globalerrors.check_session(cherrypy.session).\
                          get_workflow(wkf).get_workflow_parameters()['RequestDate'])).timetuple()
                )

            errors = globalerrors.group_errors(
                globalerrors.group_errors(errors, lambda subtask: subtask.split('/')[1],
                                          timestamp=timestamp),
                lambda workflow: globalerrors.check_session(cherrypy.session).\
                    get_workflow(workflow).get_prep_id()
                )

        # Get the names of the columns
        cols = globalerrors.check_session(cherrypy.session).\
            get_allmap()[globalerrors.get_row_col_names(pievar)[1]]

        template = lambda: GET_TEMPLATE('globalerror.html').\
            render(errors=errors,
                   columns=cols,
                   pievar=pievar,
                   acted_workflows=manageactions.get_acted_workflows(
                       serverconfig.get_history_length()),
                   readiness=globalerrors.check_session(cherrypy.session).readiness)

        try:
            return template()
        except Exception: # I don't remember what kind of exception this throws...
            time.sleep(2)
            return template()

    @cherrypy.expose
    def seeworkflow(self, workflow='', issuggested=''):
        """
        Located at ``https://localhost:8080/seeworkflow``,
        this shows detailed tables of errors for each step in a workflow.

        For the exit codes in each row, there is a link to view some of the output
        of the error message for jobs having the given exit code.
        This should help operators understand what the error means.

        At the top of the page, there are links back for :ref:`global-view-ref`,
        :ref:`show-logs-ref`, related JIRA tickets,
        and ReqMgr2 information about the workflow and prep ID.

        The main function of this page is to submit actions.
        Note that you will need to register in order to actually submit actions.
        See :ref:`new-user-ref` for more details.
        Depending on which action is selected, a menu will appear
        for the operator to adjust parameters for the workflows.

        Under the selection of the action and parameters, there is a button
        to show other workflows that are similar to the selected workflow,
        according to the :ref:`clustering-ref`.
        Each entry is a link to open a similar workflow view page in a new tab.
        The option to submit actions will not be on this page though
        (so that you can focus on the first workflow).
        If you think that a workflow in the cluster should have the same actions
        applied to it as the parent workflow,
        then check the box next to the workflow name before submitting the action.

        Finally, before submitting, you can submit reasons for your action selection.
        Clicking the Add Reason button will give you an additional reason field.
        Reasons submitted are stored based on the short reason you give.
        You can then select past reasons from the drop down menu to save time in the future.
        If you do not want to store your reason, do not fill in the Short Reason field.
        The long reason will be used for logging
        and communicating with the workflow requester (eventually).

        :param str workflow: is the name of the workflow to look at
        :param str issuggested: is a string to tell if the page
                                has been linked from another workflow page
        :returns: the error tables page for a given workflow
        :rtype: str
        :raises: cherrypy.HTTPRedirect to :ref:`global-view-ref` if a workflow
                 is not selected.
        """

        if workflow not in \
                globalerrors.check_session(cherrypy.session, can_refresh=True).return_workflows():
            raise cherrypy.HTTPRedirect('/globalerror')

        if issuggested:
            similar_wfs = []
        else:
            similar_wfs = clusterworkflows.\
                get_clustered_group(workflow, self.clusterer, cherrypy.session)

        workflowdata = globalerrors.see_workflow(workflow, cherrypy.session)

        max_error = classifyerrors.get_max_errorcode(workflow, cherrypy.session)
        main_error_class = classifyerrors.classifyerror(max_error, workflow, cherrypy.session)

        print max_error
        print main_error_class

        workflowinfo = globalerrors.check_session(cherrypy.session).get_workflow(workflow)

        drain_statuses = {sitename: drain for sitename, _, drain in \
                              sitereadiness.i_site_readiness()}

        return GET_TEMPLATE('workflowtables.html').\
            render(workflowdata=workflowdata,
                   workflow=workflow,
                   issuggested=issuggested,
                   similar_wfs=similar_wfs,
                   workflowinfo=workflowinfo,
                   params=workflowinfo.get_workflow_parameters(),
                   readiness=globalerrors.check_session(cherrypy.session).readiness,
                   mainerror=max_error,
                   acted_workflows=manageactions.get_acted_workflows(
                       serverconfig.get_history_length()),
                   classification=main_error_class,
                   drain_statuses=drain_statuses,
                   last_submitted=manageactions.get_datetime_submitted(workflow)
                  )

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def sitesfortasks(self, **kwargs):
        """
        Accessed through a popup that allows user to submit sites for workflow
        tasks that did not have any sites to run on.
        Returns operators back the :py:func:`get_action` output.

        :param kwargs: Set up in a way that manageactions.extract_reasons_params
                       can extract the sites for each subtask.
        :returns: View of actions submitted
        :rtype: JSON
        """

        manageactions.fix_sites(**kwargs)
        return self.getaction(1)

    @cherrypy.expose
    def submitaction(self, workflows='', action='', **kwargs):
        """Submits the action to Unified and notifies the user that this happened

        :param str workflows: is a list of workflows to apply the action to
        :param str action: is the suggested action for Unified to take
        :param kwargs: can include various reasons and additional datasets
        :returns: a confirmation page
        :rtype: str
        """

        cherrypy.log('args: {0}'.format(kwargs))

        if workflows == '':
            return GET_TEMPLATE('scolduser.html').render(workflow='')

        if action == '':
            return GET_TEMPLATE('scolduser.html').render(workflow=workflows[0])

        workflows, reasons, params = manageactions.\
            submitaction(cherrypy.request.login, workflows, action, cherrypy.session,
                         **kwargs)

        # Immediately get actions to check the sites list
        check_actions = manageactions.get_actions()
        blank_sites_subtask = []
        sites_to_run = {}
        # Loop through all workflows just submitted
        for workflow in workflows:
            # Check sites of recovered workflows
            if check_actions[workflow]['Action'] in ['acdc', 'recovery']:
                for subtask, params in check_actions[workflow]['Parameters'].iteritems():
                    # Empty sites are noted
                    if not params.get('sites'):
                        blank_sites_subtask.append('/%s/%s' % (workflow, subtask))
                        sites_to_run['/%s/%s' % (workflow, subtask)] = \
                            globalerrors.check_session(cherrypy.session).\
                            get_workflow(workflow).site_to_run(subtask)

        if blank_sites_subtask:
            drain_statuses = {sitename: drain for sitename, _, drain in \
                                  sitereadiness.i_site_readiness()}
            return GET_TEMPLATE('picksites.html').render(tasks=blank_sites_subtask,
                                                         statuses=drain_statuses,
                                                         sites_to_run=sites_to_run)

        return GET_TEMPLATE('actionsubmitted.html').\
            render(workflows=workflows, action=action,
                   reasons=reasons, params=params, user=cherrypy.request.login)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def getaction(self, days=0, acted=0):
        """
        The page at ``https://localhost:8080/getaction``
        returns a list of workflows to perform actions on.
        It may be useful to use this page to immediately check
        if your submission went through properly.
        This page will mostly be used by Unified though for acting on operator submissions.

        :param int days: The number of past days to check.
                         The default, 0, means to only check today.
        :param int acted: Used to determine which actions to return.
                          The following values can be used:

                          - 0 - The default value selects actions that have not been run on
                          - 1 - Selects actions reported as submitted by Unified
                          - Negative integer - Selects all actions

        :returns: JSON-formatted information containing actions to act on.
                  The top-level keys of the JSON are the workflow names.
                  Each of these keys refers to a dictionary specifying:

                  - **"Action"** - The action to take on the workflow
                  - **"Reasons"** - A list of reasons for this action
                  - **"Parameters"** - Changes to make for the resubmission
                  - **"user"** - The account name that submitted that action

        :rtype: JSON
        """
        acted = int(acted)
        if acted < 0:
            acted = None
        if acted > 1:
            acted = 1

        return manageactions.get_actions(int(days), acted=acted)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    def reportaction(self):
        """
        A POST request to ``https://localhost:8080/reportaction``
        tells the WorkflowWebTools that a set of workflows has been acted on by Unified.
        The body of the POST request must include a JSON with the passphrase
        under ``"key"`` and a list of workflows under ``"workflows"``.

        An example of making this POST request is provided in the file
        ``WorkflowWebTools/test/report_action.py``,
        which relies on ``WorkflowWebTools/test/key.json``.

        :returns: Just the phrase 'Done', no matter the results of the request
        :rtype: str
        """

        input_json = cherrypy.request.json

        if input_json['key'] == serverconfig.config_dict()['actions']['key']:
            manageactions.report_actions(input_json['workflows'])

        return 'Done'

    @cherrypy.expose
    def explainerror(self, errorcode='0', workflowstep='/'):
        """Returns an explaination of the error code, along with a link returning to table

        :param str errorcode: The error code to display.
        :param str workflowstep: The workflow to return to from the error page.
        :returns: a page dumping the error logs
        :rtype: str
        """

        workflow = workflowstep.split('/')[1]
        if errorcode == '0' or not workflow:
            return 'Need to specify error and workflow. Follow link from workflow tables.'

        errs_explained = globalerrors.check_session(cherrypy.session).\
            get_workflow(workflow).get_explanation(errorcode, workflowstep)

        return GET_TEMPLATE('explainerror.html').\
            render(error=errorcode,
                   explanation=errs_explained,
                   source=workflowstep)

    @cherrypy.expose
    def newuser(self, email='', username='', password=''):
        """
        New users can register at ``https://localhost:8080/newuser``.
        From this page, users can enter a username, email, and password.
        The username cannot be empty, must contain only alphanumeric characters,
        and must not already exist in the system.
        The email must match the domain names listed on the page or can
        be a specific whitelisted email.
        See :ref:`server-config-ref` for more information on setting valid emails.
        Finally, the password must also be not empty.

        If the registration process is successful, the user will recieve a confirmation
        page instructing them to check their email for a verification link.
        The user account will be activated when that link is followed,
        in order to ensure that the user possesses a valid email.

        The following parameters are sent via POST from the registration page.

        :param str email: The email of the new user
        :param str username: The username of the new user
        :param str password: The password of the new user
        :returns: a page to generate a new user or a confirmation page
        :rtype: str
        :raises: cherrypy.HTTPRedirect back to the new user page without parameters
                 if there was a problem entering the user into the database
        """

        if '' in [email, username, password]:
            return GET_TEMPLATE('newuser.html').\
                render(emails=serverconfig.get_valid_emails())

        add = manageusers.add_user(email, username, password,
                                   cherrypy.url().split('/newuser')[0])
        if add == 0:
            return GET_TEMPLATE('checkemail.html').render(email=email)

        raise cherrypy.HTTPRedirect('/newuser')

    @cherrypy.expose
    def confirmuser(self, code):
        """Confirms and activates an account

        :param str code: confirmation code to activate the account
        :returns: confirmation screen for the user
        :rtype: str
        :raises: A redirect the the homepage if the code is invalid
        """

        user = manageusers.confirmation(code)
        if user != '':
            return GET_TEMPLATE('activated.html').render(user=user)
        raise cherrypy.HTTPRedirect('/')

    @cherrypy.expose
    def resetpassword(self, email='', code='', password=''):
        """
        If a user forgets his or her username or password,
        navigating to ``https://localhost:8080/resetpassword`` will
        allow them to enter their email to reset their password.
        The email will contain the username and a link to reset the password.

        This page is multifunctional, depending on which parameters are sent.
        The link actually redirects to this webpage with a secret code
        that will then allow you to submit a new password.
        The password is then submitted back here via POST.

        :param str email: The email linked to the account
        :param str code: confirmation code to activate the account
        :param str password: the new password for a given code
        :returns: a webview depending on the inputs
        :rtype: str
        :raises: 404 if both email and code are filled
        """

        if not(email or code or password):
            return GET_TEMPLATE('requestreset.html').render()

        elif not (code or password):
            manageusers.send_reset_email(
                email, cherrypy.url().split('/resetpass')[0])
            return GET_TEMPLATE('sentemail.html').render(email=email)

        elif not email and code:
            if not password:
                return GET_TEMPLATE('newpassword.html').render(code=code)

            user = manageusers.resetpassword(code, password)
            return GET_TEMPLATE('resetpassword.html').render(user=user)

        raise cherrypy.HTTPError(404)

    @cherrypy.expose
    def resetcache(self):
        """
        The function is only accessible to someone with a verified account.

        Navigating to ``https://localhost:8080/resetcache``
        resets the error info for the user's session.
        It also clears out cached JSON files on the server.
        Under normal operation, this cache is only refreshed every half hour.

        :returns: a confirmation page
        :rtype: str
        """

        cherrypy.log('Cache reset by: %s' % cherrypy.request.login)

        # We want to change this directory to something set in workflowinfo soon
        for cache_file in glob.iglob('/tmp/workflowinfo/*'):
            os.remove(cache_file)

        # Force the cache reset
        if cherrypy.session.get('info'):
            cherrypy.session.get('info').teardown()
            cherrypy.session.get('info').setup()
        return GET_TEMPLATE('complete.html').render()

    @cherrypy.expose
    def listpage(self, errorcode='', sitename='', workflow=''):
        """
        This returns a list of workflows, site names, or error codes
        that matches the values given for the other two variables.
        The page can be accessed directly by clicking on a corresponding pie chart
        on :ref:`global-view-ref`.

        :param int errorcode: Error to match
        :param str sitename: Site to match
        :param str workflow: The workflow to match
        :returns: Page listing workflows, site names or errors codes
        :rtype: str
        :raises: cherrypy.HTTPRedirect to 404 if all variables are filled
        """

        if errorcode and sitename and workflow:
            raise cherrypy.HTTPError(404)

        acted = [] if workflow else \
            manageactions.get_acted_workflows(serverconfig.get_history_length())

        # Retry after ProgrammingError
        try:
            info = listpage.listworkflows(errorcode, sitename, workflow, cherrypy.session)
        except sqlite3.ProgrammingError:
            time.sleep(5)
            return self.listpage(errorcode, sitename, workflow)

        return GET_TEMPLATE('listworkflows.html').render(
            workflow=workflow,
            errorcode=errorcode,
            sitename=sitename,
            acted_workflows=acted,
            info=info)


def secureheaders():
    """Generates secure headers for cherrypy Tool"""
    headers = cherrypy.response.headers
    headers['Strict-Transport-Security'] = 'max-age=31536000'
    headers['X-Frame-Options'] = 'DENY'
    headers['X-XSS-Protection'] = '1; mode=block'
    headers['Content-Security-Policy'] = "default-src='self'"

CONF = {
    'global': {
        'server.socket_host': serverconfig.host_name(),
        'server.socket_port': serverconfig.host_port(),
        'log.access_file': 'access.log',
        'log.error_file': 'application.log'
        },
    '/': {
        'error_page.401': GET_TEMPLATE('401.html').render,
        'error_page.404': GET_TEMPLATE('404.html').render,
        'tools.staticdir.root': os.path.abspath(os.getcwd()),
        'tools.sessions.on': True,
        'tools.sessions.secure': True,
        'tools.sessions.httponly': True,
        },
    '/static': {
        'tools.staticdir.on': True,
        'tools.staticdir.dir': './static'
        },
    }

if os.path.exists('keys/cert.pem') and os.path.exists('keys/privkey.pem'):
    cherrypy.tools.secureheaders = \
        cherrypy.Tool('before_finalize', secureheaders, priority=60)
    cherrypy.config.update({
        'server.ssl_certificate': 'keys/cert.pem',
        'server.ssl_private_key': 'keys/privkey.pem'
        })

if __name__ == '__main__':

    CONF['/submitaction'] = {
        'tools.auth_basic.on': True,
        'tools.auth_basic.realm': 'localhost',
        'tools.auth_basic.checkpassword': manageusers.validate_password
        }
    for key in ['/cluster', '/resetcache', '/sitesfortasks']:
        CONF[key] = CONF['/submitaction']

    cherrypy.quickstart(WorkflowTools(), '/', CONF)

elif 'mod_wsgi' in sys.modules.keys():

    cherrypy.config.update({'environment': 'embedded'})
    application = cherrypy.Application(WorkflowTools(), script_name='/', config=CONF)
