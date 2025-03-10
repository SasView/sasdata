# test GET
# list datasets - include public and owned, disinclude private unowned
# get one dataset - succeeds if public or owned, fails if private and unowned
# get list of people with access to dataset (owner)
# fail to get list of people with access to dataset (not owner)

# test POST
# create a public dataset
# create a private dataset
# can't create an unowned private dataset

# test PUT
# edit an owned dataset
# can't edit an unowned dataset
# change access to a dataset

# test DELETE
# delete an owned private dataset
# can't delete an unowned dataset
# can't delete a public dataset
