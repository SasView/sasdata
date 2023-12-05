# Minimal makefile for Sphinx documentation
#

# Environment variables
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
APIOPTS       = -d 8 -H SasData
SOURCEDIR     = source
BUILDDIR      = build
CD            = cd

ifdef ComSpec
    RMDIR   = rmdir /s/q
    MKDIR   = mkdir
    PATHSEP = $(strip \)
else
    RMDIR   = rm -rf
    MKDIR   = mkdir -p
    PATHSEP = $(strip /)
endif

LIBDIR        = $(BUILDDIR)$(PATHSEP)lib

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	-$(RMDIR) "$(SOURCEDIR)$(PATHSEP)dev$(PATHSEP)generated/" "$(BUILDDIR)"

dir:
	-$(MKDIR) "$(BUILDDIR)"
	-$(MKDIR) "$(LIBDIR)"

.PHONY: help Makefile

# Generate the api docs
api:
	$(SPHINXAPIDOC) -o "$(SOURCEDIR)$(PATHSEP)dev$(PATHSEP)generated/" $(APIOPTS) "$(LIBDIR)"

html: dir api
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
