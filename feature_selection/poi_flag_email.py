#!/usr/bin/python


def getToFromStrings(f):
    '''
    The imported reader.py file contains functions that we've created to help
    parse e-mails from the corpus. .getAddresses() reads in the opening lines
    of an e-mail to find the To: From: and CC: strings, while the
    .parseAddresses() line takes each string and extracts the e-mail addresses
    as a list.
    '''

    all_strings = str(f).split("X-From")[0].split(": ")

    from_string = ""
    to_string = ""
    cc_string = ""
    for i in range(len(all_strings)):
        if from_string == "" and all_strings[i][-4:] == "From":
            from_string = all_strings[i+1].split(", ")
        elif to_string == "" and all_strings[i][-2:] == "To":
            to_string = all_strings[i+1].split(", ")
        elif cc_string == "" and all_strings[i][-2:] == "Cc":
            cc_string = all_strings[i+1].split(", ")
            break

    to_emails = [j for i in [i.split() for i in to_string] for j in i][:-1]
    from_emails = [j for i in [i.split() for i in from_string] for j in i][:-1]
    cc_emails = [i.split() for i in cc_string][:-1]

    return to_emails, from_emails, cc_emails



### POI flag an email

def poiFlagEmail(f):
    """ given an email file f,
        return a trio of booleans for whether that email is
        to, from, or cc'ing a poi """
    import sys
    sys.path.append("../final_project/")
    from poi_email_addresses import poiEmails
    to_emails, from_emails, cc_emails = getToFromStrings(f)

    ### poi_emails.poiEmails() returns a list of all POIs' email addresses.
    poi_email_list = poiEmails()

    to_poi = False
    from_poi = False
    cc_poi = False

    ### to_poi and cc_poi are boolean variables which flag whether the email
    ### under inspection is addressed to a POI, or if a POI is in cc,
    ### respectively. You don't have to change this code at all.

    ### There can be many "to" emails, but only one "from", so the
    ### "to" processing needs to be a little more complicated
    if to_emails:
        ctr = 0
        while not to_poi and ctr < len(to_emails):
            if to_emails[ctr] in poi_email_list:
                to_poi = True
            ctr += 1
    if cc_emails:
        ctr = 0
        while not cc_poi and ctr < len(cc_emails):
            if cc_emails[ctr] in poi_email_list:
                cc_poi = True
            ctr += 1

    #################################
    ######## your code below ########
    ### set from_poi to True if #####
    ### the email is from a POI #####
    #################################
    if from_emails:
        ctr = 0
        while not from_poi and ctr < len(from_emails):
            if from_emails[ctr] in poi_email_list:
                from_poi = True
            ctr += 1

    #################################
    return to_poi, from_poi, cc_poi
