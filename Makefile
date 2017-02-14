format:
	go fmt

test-all:
	go test -v

test-all-trace:
	go test -v -tracecost

# test ExampleF_xorbits (for example), and keep checking gradients while running the test
# warning: may be a bit slow, not in this but in other cases
test-check-grad:
	go test -run xor -tracecost -checkgrad

# show command-line options
help-cli:
	go run cmd/main.go --help || true

coverage:
	go test -coverprofile=coverage.out
	go tool cover -html="coverage.out"

lint: format
	go get -u github.com/golang/lint/golint
	golint | grep -v underscore

default: lint test
