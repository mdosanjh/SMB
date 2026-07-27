# -*- Makefile -*-
#
# Copyright 2006 Sandia Corporation. Under the terms of Contract
# DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government
# retains certain rights in this software.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA  02110-1301, USA.


RELEASE_VERSION = 1.0

TARGET = msgrate
SRC    = msgrate_kokkos.cpp
OBJ    = msgrate_kokkos.o

# Use the MPI C++ compiler wrapper.
# You may override this on the command line, e.g.
#   make CXX=mpicxx
#   make CXX=mpiCC
#   make CXX=CC
CXX = mpicxx

CPPFLAGS = -DVERSION=$(RELEASE_VERSION)
CXXFLAGS = -O3 -std=c++20
LDFLAGS  =
LDLIBS   =

# Path to an installed Kokkos.
# Override on the command line, for example:
#   make KOKKOS_ROOT=/path/to/kokkos/install
KOKKOS_ROOT ?= /path/to/kokkos/install

CPPFLAGS += -I$(KOKKOS_ROOT)/include

# Some Kokkos installs use lib, others use lib64.
LDFLAGS  += -L$(KOKKOS_ROOT)/lib
LDFLAGS  += -L$(KOKKOS_ROOT)/lib64

# Optional runtime library search paths.
# Uncomment these if your environment does not already set LD_LIBRARY_PATH.
# LDFLAGS += -Wl,-rpath,$(KOKKOS_ROOT)/lib
# LDFLAGS += -Wl,-rpath,$(KOKKOS_ROOT)/lib64

# For many Kokkos installs, kokkoscore is sufficient for this code.
# Some installs also provide/use kokkoscontainers and kokkossimd.
LDLIBS += -lkokkoscontainers -lkokkoscore -ldl

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(OBJ): $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET) *~

dist:
	rm -rf msgrate-$(RELEASE_VERSION)
	mkdir msgrate-$(RELEASE_VERSION)
	(for file in msgrate_kokkos.cpp README Makefile GPL.txt LICENSE.txt ; do \
	cp $$file msgrate-$(RELEASE_VERSION)/. ; \
	chmod 644 msgrate-$(RELEASE_VERSION)/$$file ; \
	done)
	tar czf msgrate-$(RELEASE_VERSION).tar.gz msgrate-$(RELEASE_VERSION)
	rm -rf msgrate-$(RELEASE_VERSION)

.PHONY: clean dist
