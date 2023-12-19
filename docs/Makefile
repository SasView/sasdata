# Minimal makefile for Sphinx documentation

# Environment variables
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
APIOPTS       = -d 8 -H SasData
DOCSDIR       = docs
SOURCEDIR     = source
BUILDDIR      = build
DEVDIR        = dev
LIBDIR        = lib
GENDIR        = generated
UP            = ..

ifdef ComSpec
    RMDIR   = rmdir /s/q
    MKDIR   = mkdir
    PATHSEP = $(strip \)
else
    RMDIR   = rm -rf
    MKDIR   = mkdir -p
    PATHSEP = $(strip /)
endif

SASDATABUILD  = $(UP)$(PATHSEP)$(BUILDDIR)$(PATHSEP)$(LIBDIR)
DOCSSRC       = $(SOURCEDIR)
DEV           = $(DOCSSRC)$(PATHSEP)$(DEVDIR)
DEVGEN        = $(DEV)$(PATHSEP)$(GENDIR)
DOCSBUILD     = $(DOCSSRC)$(PATHSEP)$(BUILDDIR)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	-$(RMDIR) "$(DEVGEN)" "$(DOCSBUILD)"

dir:
	-$(MKDIR) "$(DEVGEN)"
	-$(MKDIR) "$(DOCSBUILD)"

.PHONY: help Makefile

# Generate the api docs
api:
	$(SPHINXAPIDOC) -o "$(DEVGEN)" $(APIOPTS) "$(SASDATABUILD)"

html: dir api
	@$(SPHINXBUILD) -M $@ "$(DOCSSRC)" "$(DOCSBUILD)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
