.PHONY: generate profile clear

generate:
	@python3 -m numneur.setup

force:
	@make clear
	@python3 -m numneur.setup --force_build

profile:
	@make clear
	@python3 -m numneur.setup --profile

notrace:
	@make clear
	@python3 -m numneur.setup --notrace

hardcore:
	make clear
	@python3 -m numneur.setup --hardcore

hardcoreprofile:
	make clear
	@python3 -m numneur.setup --hardcore --profile

clear:
	@echo "Cleaning all.."
	@rm -f numneur/*.c
	@rm -f numneur/*.so
	@rm -f numneur/*.html
	@rm -R -f numneur/build
	@rm -R -f numneur/__pycache__
	@echo "Cleaned."