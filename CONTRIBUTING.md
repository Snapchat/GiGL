# Contributing to GiGL

Thank you for your interest in GiGL! We welcome community contributions and appreciate your time and effort in helping
improve the project. Before getting started, please take a moment to review these guidelines.

## Code of Conduct

We want this project to be a welcoming space for everyone. By contributing, you agree to follow our
[Code of Conduct](CODE_OF_CONDUCT.md) and help keep the community respectful and inclusive.

## Reporting Issues

**If security related please see [SECURITY.md](SECURITY.md) for guidance.**

If you find a bug or have a feature request, please open an issue and provide as much detail as possible:

- Search existing issues to avoid duplicates.
- Clearly describe the issue with steps to reproduce (If applicable).
  - Include relevent specs used both task and resource.
  - Provide relevant logs or screenshots if applicable.
- Expected and actual behahavior.
- List Suggested solutions (if any).

## Legal Terms

By submitting a contribution, you represent and warrant that:

- It is your original work, or you have sufficient rights to submit it.
- You grant the GiGL maintainers and users the right to use, modify, and distribute it under the MIT license (see
  [LICENSE](LICENSE) file); and
- To the extent your contribution is covered by patents, you grant a perpetual, worldwide, non-exclusive, royalty-free,
  irrevocable license to the GiGL maintainers and users to make, use, sell, offer for sale, import, and otherwise
  transfer your contribution as part of the project.

We do not require a Contributor License Agreement (CLA). However, by contributing, you agree to license your submission
under terms compatible with the MIT License and to grant the patent rights described above. If your contribution
includes third-party code, you are responsible for ensuring it is MIT-compatible and properly attributed.

Moral Rights Disclaimer: Where permitted by law, you waive any moral rights (e.g., the right to object to modifications)
in your contribution. If such rights cannot be waived, you agree not to assert them in a way that interferes with the
project’s use of your contribution.

## Open Development

We follow an open development process where both core team members and the community contribute through the same review
process. All pull requests, regardless of the author, go through the same review and approval workflow to ensure
consistency and quality.

## How to Contribute

### Proposing a Non-Trivial Change

- Before starting major work, open an issue to discuss your proposal with the maintainers.
- Clearly outline the problem and your proposed solution.
- Gather feedback and refine your approach before implementation.
- This ensures alignment with project goals and avoids unnecessary work.

### Submitting Code

1. Fork the repository and create a feature branch.
1. Ensure all unit tests pass before submitting.
1. Add relevant unit/integration/performance tests.
1. Submit a pull request (PR) with a clear description of your changes.
1. Address review feedback promptly.
1. All changes should be submitted to the `main` branch via a pull request.

### Semantic Versioning & Changelog

We adhere to [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH) to ensure clear version tracking:

- **MAJOR** versions introduce breaking changes.
- **MINOR** versions add functionality in a backward-compatible manner.
- **PATCH** versions fix bugs and small issues.

All significant changes are recorded in the [CHANGELOG](CHANGELOG.md), where contributors should document major updates,
new features, and fixes.

TODO: (svij) More instructions to come on how release process will be managed.

### Commit Guidelines

- We squash commit PRs. Ensure your PRs follow the [pull_request_template](pull_request_template.md).

### PR Checklist:

#### Code Correctness

- Is the code logically correct?
- Are there any edge cases that we have not covered?
- Has the author executed on a reasonable testing plan and/or has the reviewer tested the changes themselves?
- Does the PR meet its objective of satisfying the task requirements? i.e. will it scale to necessary requirements

#### Code Comprehension/Consistency

- Will the change be easily understandable to a broader audience?
- Will the solution make sense as the code-base evolves?
- Does the PR follow agreed upon/ industry best practices, and follow patterns already established in the codebase?

### Author’s Responsibility:

#### Most important: Create *[Small PRs](https://google.github.io/eng-practices/review/developer/small-cls.html)*

Explicitly tag two people on your PR (See [OWNERS](OWNERS) for list of reviewers). Generally the first review to your PR
should be done within 1-2 business days depending on scope from when you ask for review; if this isn't done, it is your
responsibility to follow up for a response or to find a different reviewer if needed using our
[Communication Channels](#questions). In cases when the PR is not “small”
([see what it means for PR to be small](https://google.github.io/eng-practices/review/developer/small-cls.html#what_is_small),
1-2 business days guidance is not reasonable, and reviewers may push back and ask you to break the PR down into “small
PRs.”

More generally,

- In rare cases, you may need to add more reviewers in cases of driving consensus or leveraging certain “domain
  expertise”.
- In some “contextually” very small PRs, you may only require one reviewer, for example:
  - Adding a new small unit test.
  - Formatting, variable or directory name change; this could span many files and lines; contextually, it is still a
    “very small” change.
  - Editing the OWNERS file
  - Fixing a spelling mistake or adding a few lines to a README

Happy coding!

## Questions/Comments/Ideas?

If you need help, or need to get in touch with the primary maintainers of the project, please open a discussion/issue.
