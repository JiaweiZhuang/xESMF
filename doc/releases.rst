Release how-to
==============

Here are the step by step instructions to release a new version of xESMF.

#. Make sure :file:`CHANGES.rst` is up to date and includes a section on the version to be released;
#. On GitHub, go the Releases_ page and click on :guilabel:`Draft a new release`;
#. Enter new version in :guilabel:`Tag version` (e.g. v<major>.<minor>.<patch>);
#. Enter the :guilabel:`Release title` (e.g. the same tag);
#. Copy the relevant section of :file:`CHANGES.rst` in the description;
#. Click :guilabel:`Publish release`; Keep this tab open, we'll come back to it.
#. Go to Actions_ and in the left-hand menu select :guilabel:`Upload xesmf to PyPi`;
#. Click on the release you just made;
#. At the bottom of the page, wait for the action to complete then download the :guilabel:`artifact` to your local disk;
#. Unzip it;
#. Run shell command ``sha256sum`` on `xesmf-<version>.tar.gz` and copy the checksum string;
#. Add the artifacts (drag & drop) to the :guilabel:`Assets` of the Github release you just created.
#. Go to the conda-forge repo and edit the `recipe/meta.yml <https://github.com/conda-forge/xesmf-feedstock>`_ file;
#. Update the version on the first line to the latest release;
#. Update the `sha256` value in the `source` section to the checksum just calculated;
#. Enter a commit message, e.g. (Update version to <version>);
#. Submit the pull request (:guilabel:`Propose changes`);
#. Get an approval from a maintainer to merge the pull request.

Note that Conda Forge's Continuous Integration (CI) checks will fail if you haven't uploaded the artifacts to the Github release assets or if the sha256 checksum does not match the asset's. In case you need to re-run the CI checks, just post the comment `@conda-forge-admin, please restart ci`.

.. _Releases: https://github.com/pangeo-data/xESMF/releases
.. _Actions: https://github.com/pangeo-data/xESMF/actions
