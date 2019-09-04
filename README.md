# manager_option_library (mo)
This is a library containing a general hierarchical RL strategy that aims at being used in any kind of environment.

# How it works ?
Add to your requirements.txt:
```
git+https://git@github.com/damienAllonsius/manager_option_library
```
Import `mo` and get the option framework power ! You only need to implement policies for options.
Specify your choices by creating a classes inheriting from `AbstractManager` and `ÀbstractOption`.

# What it does
(see `_train_simulate_manager` function in `ÀbstractManager` class)

First, the Manager selects a new Option through its policy inheriting from `AbstractPolicyManager` (at the very begining, it returns an exploration option - an instance of a class inheriting from `AbstractOptionExplore` - since no option is discovered yet).

Then the Option acts using its own poliy (inheriting from `AbstractPolicyOption`) and the Manager checks whether if the Option's mission is done or not. It it's done, then the Manager checkes whether or not it's done __correctly__. The option is updated according to these conditions.

If the Option is not done yet then it continues acting. Otherwise the Manager updates its (abstract) representation of the environment and selects a new Option.

The process continues until the Manager is done.

# FAQ
__Q__ : How does the manager selects a new option ? What is the abstract representation ? <br/>
__A__ : *Secret sauce.*


__Q__ : How does the manager know if the Option is done ? <br/>
__A__ : *The manager compares the last abstract state recorded to the abstract state returned by the environment. It returns True if and only if those states are different.*

__Q__ : How does the manager know if the new abstract state is the correct one ? <br/>
__A__ : *It compares the new abstract state the the abstract state selected initially when it activated the current Option.*
