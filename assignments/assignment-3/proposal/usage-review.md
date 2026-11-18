# N24News usage-permission review

**Checked:** 18 November 2026 · **Decision:** Not cleared for the proposed ML experiments

## Findings

| Source inspected | Finding |
|---|---|
| [Official N24News source](https://github.com/billywzh717/N24News/tree/047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106) | Provides metadata, reconstruction code, a paper citation, and a dataset download link; no standalone dataset license or rights-holder permission grant was found. |
| Current upstream file tree and public issues | No license file or explicit research-use grant was found in the inspected material. |
| [NYT Terms of Service](https://help.nytimes.com/115014893428-Terms-of-Service), §2.1 | The stated personal/non-commercial use does not include ML/AI development uses without prior written consent. Academic/non-commercial status is not itself a grant. |
| Same terms, §4.1 | Specifies prior written consent for the listed automated collection and ML/AI development uses. |
| [NYT Licensing](https://nytlicensing.com/contact/) | Provides a rights-holder contact route; a contact page is not permission. |

The publisher's official terms and licensing pages were retrieved successfully by HTTP on the review date. Findings above paraphrase those sources; no legal exception or institutional license was established. The paper's publication license does not license the underlying news text/photos, and a mirror's license label does not establish that the mirror controls those rights.

**No permission has been supplied by the group.** No request was sent, and no grant, approval, or clearance is claimed. Instructor approval is distinct from rights-holder permission. The completed 18-pair preliminary inspection does not establish authorization to train or publish source imagery.

To close this issue, retain an applicable written permission/license reference and scope, or replace N24News with a dataset whose published terms cover the intended use. Until then, do not mark the licensing requirement resolved or start main experiments. This is a recorded dataset-selection limitation, not an assertion about an automatic grade penalty.

## Permission-request draft — not sent

**Recipient:** NYT Licensing, via the contact page above<br>
**Subject:** Permission request: CO3133 academic image–text classification coursework

We are Group MobyDick, four students at Ho Chi Minh City University of Technology – VNU-HCM, taking CO3133 Deep Learning and Its Applications. We seek permission to use a 7,500-pair subset of N24News, comprising original NYT headlines and associated images in six editorial categories, for non-commercial assessed coursework.

The proposed work would download/store the selected pairs locally, preprocess the images and headlines, extract features using frozen pretrained image/text encoders, train classification and fusion heads, and evaluate the results. We plan to publish source code, identifiers/hashes, aggregate metrics, and reconstruction instructions. We would not redistribute the raw dataset.

Please confirm whether you can authorize these uses, any limits on storage/access or retention, and whether third-party photographs require separate clearance. Please also state whether a small number of attributed qualitative examples may appear in the public course report, slides, GitHub Pages website, and YouTube presentation, and whether publication of trained classifier heads is covered. We will not infer these permissions from public dataset availability.

Please provide applicable written terms and required attribution. No training-use or example-publication permission is presumed by this request.
