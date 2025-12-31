---
title: Ex-L8 Faang Manager's Notes on Code Generation Tools
---

I wanted to write down some raw thoughts on using Claude Code as 2025 come to an end. I've been using these tools for a while now and have seen similar patterns when I talk to folks. A lot of these ideas are still working themselves out in terms of practice, and I wanted to write some of them down before the next wave of model improvements.

For a slaphappy introduction to CC checkout this [guide](https://sankalp.bearblog.dev/my-experience-with-claude-code-20-and-how-to-get-better-at-using-coding-agents/) from my moot [dejavucoder](https://twitter.com/dejavucoder). 

----
>Note #1: Don't outsource your thinking to a model, it may or may not be aligned to what you want to do.

If you started using code generation models from around gpt 3.5, you'd be accustomed to a chat interface flow, where you type something into the chat window and then transfer over the generated code into your IDE or editor. You'd also be familiar with tools like Cursor that offer both in-line and file-level completion. 

I've personally only liked using models that generate snippets that solve a problem or search documentation to solve a problem. I've always found it more responsible to outsource some amount of grunt work to a model but not the thinking itself.

>Note #2: Context rot is key to manage. It is tempting to stuff the context as much as you can, but this will increase the chance of running into hallucinations or bugs, or worse, sycophancy.

Chat models offer a sense of statelessness, they might have caching and memory now, but when you start a new chat the expectation is that they are processing instruments that are biased with every input you give. Be prepared to clean out the context as needed.

>Note #3: Outsource grunt work that is predictable and consistent in the task shape.

Chat models are amazing at dealing with fuzziness, if you pass in text and ask it to spot specific strings, they will do a good job. You don't need to use a vector database to do this ask if you're think about "but what about matching semantically similar terms"? The model is doing the vector search for you over some context window.

>Note #4: The harness is what guides the model. Both the model and the harness must be good for the Developer to feel magic.

The obvious limitation with chat models is that they introduce friction into the developer workflow. While the context switching aspect is one part, the friction is copying over code and papering over the rough edges into your IDE. While not exactly a first-world problem it is annoying and hits productivity.

This is where Claude Code comes in with its Agents approach. While CC is a wrapper around Claude, the Harness (ie. the prompts) and the model tuning are substantial. You may or may not have the same experience with other Harnesses like OpenAI Codex or OpenHands.

CC (esp. Opus 4.5 now) is amazing in terms of general intelligence. Pure prompting, with no md files or any other indexes, or additional domain context, *will increase* your productivity. The [Anthropic blogs](https://claude.com/blog) for CC are a good starting point to understand where things are.

>Note #5: If you don't know what you don't know, you might not be able to ask the model the right questions.

You can do more than just pure prompting and CC along with other tools have come a long way in just this year. I would recommend setting up a [CLAUDE.md](https://claude.com/blog/using-claude-md-files) file using CC cli's `init` command. This will go over your codebase and set up basic instructions. As noted in the blog, you should add things to the md file as and when you encounter a situation that has a known repeatable action. For example, if you have a typescript project and you use `pnpm` with `vitest` you might write that down in the md file. 

You might also encounter some very interesting issues. For whatever reason, Opus 4.5 doesn't think `vitest` `clearMocks` and `resetMocks` can lead to incorrect unit test setup. This is a big enough issue that other languages/frameworks have strong opinions about how to write tests, or setup Test Driven Development or Spec Driven Development. Adding them as you encounter them to the md file helps.

>Note #6: Grounding the model remains a key aspect, regardless of how the model is exposed to you as a user.

The md file is an entry point for CC that helps ground the model and the harness. Having consistent themes there will help CC do a better job. It might not work if you have multiple personas there. I have a `How does a CTO think` section in my md file that guides high level design choices, including a simple "please structure your changes at a high level before we start working on it."

>Note #7: Ask the model to plan first. If you don't have a plan you will end up with slop because you can't think through a plan that doesn't exist.

I always ask CC to generate an implementation plan as part of the md file. This helps keep track of tasks and offers a checkpoint to the work done. If for some reason you need to start over, at the very least you have a TODO list that can be executed one after the other.

>Note #8: Subagents might take some effort to set up and might not be worth it for your personal productivity. It might make sense to have 1 agent per team member and cut down the overall team size.

I don't use CC with multiple agents. The analogy of a direct report and an agent is apt, if you have 10 sub-agents, that's 10 sources of potential failure. Typically people use CC to spawn an inspector or manager agent that will "manage" employees. This may or may not work, and when it doesn't work you have an explosion in the number of combinations you have to think through to understand why it failed. I'm a fan of the [two pizza teams](https://aws.amazon.com/executive-insights/content/amazon-two-pizza-team/) idea, and this core principle has to apply to agents too. This shortcoming will change within the next year.

>Note #9: Token consumption is a key metric. If a task needs too many tokens to get an output there is an opportunity to optimize.

I also haven't set up CC with LSP or use any other semantic code retrieval setup like [Serena](https://github.com/oraios/serena). Both approaches are a bit new but will help with token consumption over time.

>Note #10: Nobody gets promoted for writing unit tests, but if you're a 2-person billion dollar company, you're going to need unit tests. Also, unit tests will never not save your life.

One funny thing I've noticed is that CC takes up a lot of tokens to analyze build outputs (pass or fail). I typically just run the build myself and paste the error, a smaller cheaper model (say via a tool call) might be the right way to automate this step.

When dealing with code, having a test suite is critical. Otherwise, as is often the case, there is too much to do and not enough human mental bandwidth, leading to regressions. Having infra that helps roll-back and roll-forward is critical and a different discussion. From the pov of CC however, having a good test suite to start with is needed to enable the agentic aspects. Without this environment setup, the agent's ability to have autonomy and self-heal is limited from the get go.

>Note #11: If you're not thinking declaratively you're going to have a hard time managing AI output.

With AI tools, the cost of greenfield test setup should be very low. If you organize your Test Driven Development correctly, you can utilize CC to write tests and vet them properly. This frame of mind will also help you write your code better. CC and other models tend to write long code blocks. Declarative programming, writing guards upfront, good error handling and logging etc etc, are still critical aspects; CC will easily and quickly set up a Feature Flag for you, you take advantage of this productivity boost. These ideas were key pre-LLM and will continue to be critical.

>Note #12: Use a combination of models for sanity checks.

I primarily use CC, but will ask Codex to run code reviews. Having some variety here is important. Ideally you could ask a new instance to do a code review, but that's like asking the PR implementor to review their PR with a fresh mind the next day, it works but that's not the point of a code review. I however make sure to check all the generations as I would with a colleague's work.

>Note #13: Treat the Agent like a Pure Function when you need to explore, have specific prompts if needed.

The final note is on Legacy codebases. CC will work real well but it might not have the right context. This is where I try to step in and guide CC like a junior engineer might encounter a large legacy codebase, try to understand the code first, make notes, read specific doc versions, write unit tests, make changes.
